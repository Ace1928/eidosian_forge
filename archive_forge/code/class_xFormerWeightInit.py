import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
class xFormerWeightInit(str, Enum):
    Timm = 'timm'
    ViT = 'vit'
    Moco = 'moco'
    Small = 'small'