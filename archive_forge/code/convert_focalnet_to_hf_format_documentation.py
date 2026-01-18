import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
Convert FocalNet checkpoints from the original repository. URL: https://github.com/microsoft/FocalNet/tree/main