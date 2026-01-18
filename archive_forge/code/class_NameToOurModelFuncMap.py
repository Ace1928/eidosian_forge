import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging
class NameToOurModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return the correct hugging face RegNet class reference.
    """

    def __getitem__(self, x: str) -> Callable[[], nn.Module]:
        if 'seer' in x and 'in1k' not in x:
            val = RegNetModel
        else:
            val = RegNetForImageClassification
        return val