import argparse
import requests
import torch
from PIL import Image
from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor
Convert Swin SimMIM checkpoints from the original repository.

URL: https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#simmim-pretrained-swin-v1-models