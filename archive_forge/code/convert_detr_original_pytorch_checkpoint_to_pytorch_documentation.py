import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor
from transformers.utils import logging

    Copy/paste/tweak model's weights to our DETR structure.
    