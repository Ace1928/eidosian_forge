import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
def _is_cuda_at_least_sm80(device: torch.device) -> bool:
    return _is_cuda() and torch.cuda.get_device_capability(device) >= (8, 0)