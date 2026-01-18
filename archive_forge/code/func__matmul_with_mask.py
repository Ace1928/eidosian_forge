import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _matmul_with_mask(a: torch.Tensor, b: torch.Tensor, mask: Optional[Union[torch.Tensor, 'SparseCS']]) -> torch.Tensor:
    if mask is None:
        return a @ b
    if _has_cpp_library and mask.dtype == torch.bool:
        if isinstance(mask, SparseCS):
            return mask.matmul_with_mask(a, b)
        if mask.is_sparse:
            mask = _broadcast_batch(mask, a.shape[0])
            mask = mask.to(dtype=a.dtype)
        return torch.ops.xformers.matmul_with_mask(a, b, mask)
    if _has_cpp_library:
        assert not isinstance(mask, SparseCS)
    att = a @ b
    if mask.dtype == torch.bool:
        assert not isinstance(mask, SparseCS)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        att[~mask] = float('-inf')
    else:
        if not isinstance(mask, SparseCS) and mask.ndim == 3 and (mask.shape[0] != att.shape[0]) and (att.shape[0] % mask.shape[0] == 0):
            repeat_factor = att.shape[0] // mask.shape[0]
            mask = mask.repeat([repeat_factor, 1, 1])
            logger.info('Mismatched batch dimensions for mask, repeating mask.')
        att += mask
    return att