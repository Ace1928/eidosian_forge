import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessor, Swinv2Config
from transformers.utils import logging

    Copy/paste/tweak model's weights to our DPT structure.
    