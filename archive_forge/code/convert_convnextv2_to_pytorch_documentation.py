import argparse
import json
import os
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging

    Copy/paste/tweak model's weights to our ConvNeXTV2 structure.
    