from torchvision import transforms, datasets
from torchvision.utils import save_image

import torch
import os

torch.manual_seed(42)
# Paths to your train and val directories
train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")
data_transforms = transforms.Compose([transforms.ToTensor()])
# Create a pytorch dataset from a directory of images
train_dataset = datasets.ImageFolder(train_dir, data_transforms)
val_dataset = datasets.ImageFolder(val_dir, data_transforms)

for i, (img, label) in enumerate(train_dataset):
	if i == 0 or label != prev_label:
		os.mkdir(f'{train_dir}/{label}')
	save_image(img, f'{train_dir}/{label}/{i}__.png')
	prev_label = label

for i, (img, label) in enumerate(val_dataset):
	if i == 0 or label != prev_label:
		os.mkdir(f'{val_dir}/{label}')
	save_image(img, f'{val_dir}/{label}/{i}__.png')
	prev_label = label
