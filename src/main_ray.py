import ray
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
import os

from getBadnet import PoisonedTestLoader
from getSaliency import GradCAMLoader
from calASR import ReconEvaluator
from vae import FlexibleVAE

ray.init(address="auto")  # 连接已有集群

MODEL_CKPT = "/home/ray/raybackdoorv1/resnet18_badnet_epoch40.pth"
VAE_CKPT = "/home/ray/raybackdoorv1/vaemodel/datacifar10_latent1024_epoch1200.pth"
assert os.path.exists(MODEL_CKPT)
assert os.path.exists(VAE_CKPT)

@ray.remote
class PoisonedReconActorCPU:
    def __init__(self, model_ckpt, vae_ckpt, latent_dim=1024, input_size=32):
        self.device = 'cpu'
        # 分类模型
        num_classes = 10
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
        self.vae.eval()

    def run_pipeline(self, test_subset_indices, dataset_root="/home/ray/raybackdoorv1/data",
                     target_label=0, trigger_size=3,
                     gradcam_layer='layer3', gradcam_threshold=0.8, batch_size=8):

        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])
        # 原始 testset
        full_testset = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
        # 子集
        testset = Subset(full_testset, test_subset_indices)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Poisoned 测试集
        poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size,
                                            target_label=target_label, batch_size=batch_size)
        poi_testloader = poi_loader_obj.get_poi_testloader()

        # Grad-CAM mask
        gradcam_loader_obj = GradCAMLoader(self.model, poi_testloader,
                                          device=self.device, target_layer_name=gradcam_layer,
                                          threshold=gradcam_threshold)
        gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

        # VAE 重建 + mask 融合
        recon_evaluator = ReconEvaluator(self.model, self.vae, gradcam_testloader,
                                         device=self.device, batch_size=batch_size)
        acc = recon_evaluator.test_accuracy()
        return acc

# ---------------------
# 测试集拆分
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.247, 0.243, 0.261])
])
full_testset = torchvision.datasets.CIFAR10(root="/home/ray/raybackdoorv1/data", train=False, download=True, transform=transform)
num_actors = 6  # 你希望创建的 Actor 数量
dataset_len = len(full_testset)
chunk_size = dataset_len // num_actors

# 分配每个 Actor 的索引
subsets_indices = [
    list(range(i*chunk_size, (i+1)*chunk_size if i < num_actors-1 else dataset_len))
    for i in range(num_actors)
]

# 初始化 Actor
actors = [
    PoisonedReconActorCPU.options(num_cpus=1, memory=0.8*1024*1024*1024).remote(
        model_ckpt=MODEL_CKPT,
        vae_ckpt=VAE_CKPT,
        latent_dim=1024,
        input_size=32
    )
    for _ in range(num_actors)
]

# 并行运行 pipeline
futures = [actor.run_pipeline.remote(indices) for actor, indices in zip(actors, subsets_indices)]
results = ray.get(futures)

# 汇总结果（加权平均或直接平均）
overall_acc = sum(results)/len(results)
for i, acc in enumerate(results):
    print(f"Actor {i} accuracy:", acc)
print("Overall accuracy:", overall_acc)
