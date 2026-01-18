import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        