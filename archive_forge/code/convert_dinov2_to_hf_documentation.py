import argparse
import json
from pathlib import Path
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging

    Copy/paste/tweak model's weights to our DINOv2 structure.
    