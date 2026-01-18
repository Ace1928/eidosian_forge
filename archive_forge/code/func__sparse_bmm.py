import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
        Batch matrix multiply between a sparse matrix and a dense matrix
        """
    assert a.ndim == b.ndim == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    return SparseBMM.apply(a, b)