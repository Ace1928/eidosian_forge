import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _create_random_sparsity(matrix, sparsity, divisible_by=4):
    assert matrix.ndim == 3
    keep = torch.rand_like(matrix[0], dtype=torch.float32) > sparsity
    nonzero = torch.nonzero(keep)
    nnz = nonzero.shape[0]
    nonzero = nonzero[:nnz - nnz % divisible_by]
    i, j = nonzero.unbind(1)
    output = torch.zeros_like(matrix)
    bdim = torch.arange(matrix.shape[0], device=matrix.device)[:, None]
    output[bdim, i, j] = matrix[bdim, i, j]
    return output