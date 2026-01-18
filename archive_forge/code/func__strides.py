import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
def _strides(x: Optional[torch.Tensor], *stride_names: str):
    if x is None:
        return {f'stride_{name}': None for name in stride_names}
    assert x.ndim == len(stride_names)
    return {f'stride_{name}': s for name, s in zip(stride_names, x.stride())}