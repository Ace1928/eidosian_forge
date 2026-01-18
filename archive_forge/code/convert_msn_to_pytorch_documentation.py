import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
Convert ViT MSN checkpoints from the original repository: https://github.com/facebookresearch/msn