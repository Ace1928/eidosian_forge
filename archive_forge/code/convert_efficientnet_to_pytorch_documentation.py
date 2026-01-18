import argparse
import json
import os
import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our EfficientNet structure.
    