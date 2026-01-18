import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _post_process_flash_output(out: torch.Tensor, og_size):
    if not out.is_nested and out.size(-1) != og_size:
        out = out[..., 0:og_size]
    return out