import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import PvtImageProcessor, PvtV2Config, PvtV2ForImageClassification
from transformers.utils import logging

    Copy/paste/tweak model's weights to our PVT structure.
    