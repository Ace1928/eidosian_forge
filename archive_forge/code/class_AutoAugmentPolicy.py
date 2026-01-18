import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from . import functional as F, InterpolationMode
class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    IMAGENET = 'imagenet'
    CIFAR10 = 'cifar10'
    SVHN = 'svhn'