import argparse
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as F
from transformers import DetrImageProcessor, ResNetConfig, TableTransformerConfig, TableTransformerForObjectDetection
from transformers.utils import logging

    Copy/paste/tweak model's weights to our DETR structure.
    