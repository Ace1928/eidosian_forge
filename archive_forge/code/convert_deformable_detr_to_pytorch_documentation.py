import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from transformers.utils import logging

    Copy/paste/tweak model's weights to our Deformable DETR structure.
    