import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import (
from transformers.image_utils import PILImageResampling
from transformers.utils import logging

    Copy/paste/tweak model's weights to our ViT structure.
    