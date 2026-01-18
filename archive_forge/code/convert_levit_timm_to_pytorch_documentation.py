import argparse
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path
import timm
import torch
from huggingface_hub import hf_hub_download
from transformers import LevitConfig, LevitForImageClassificationWithTeacher, LevitImageProcessor
from transformers.utils import logging
Convert LeViT checkpoints from timm.