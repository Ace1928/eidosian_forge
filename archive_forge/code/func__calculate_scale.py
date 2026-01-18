import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _calculate_scale(query, scale):
    softmax_scale = scale if scale is not None else math.sqrt(1.0 / query.size(-1))
    return softmax_scale