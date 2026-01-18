import argparse
import requests
import torch
from PIL import Image
from transformers import SegGptConfig, SegGptForImageSegmentation, SegGptImageProcessor
from transformers.utils import logging
Convert SegGPT checkpoints from the original repository.

URL: https://github.com/baaivision/Painter/tree/main/SegGPT
