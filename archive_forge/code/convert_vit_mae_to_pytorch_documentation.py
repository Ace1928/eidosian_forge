import argparse
import requests
import torch
from PIL import Image
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEImageProcessor
Convert ViT MAE checkpoints from the original repository: https://github.com/facebookresearch/mae