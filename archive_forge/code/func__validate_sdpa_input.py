import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _validate_sdpa_input(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, dropout_p=0.0, is_causal=False, scale=None):
    if not isinstance(query, NestedTensor) or not isinstance(key, NestedTensor) or (not isinstance(value, NestedTensor)):
        raise ValueError(f'Expected query, key, and value to be nested tensors, but got query.is_nested: {query.is_nested}, key.is_nested: {key.is_nested}, and value.is_nested: {value.is_nested} instead.')
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(f'Expected query, key, and value to have the same dtype, but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, and value.dtype: {value.dtype} instead.')
    if query.device != key.device or query.device != value.device:
        raise ValueError(f'Expected query, key, and value to have the same device type, but got query.device: {query.device}, key.device: {key.device}, and value.device: {value.device} instead.')
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError(f'Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: {query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead.')
    if query._ragged_idx != key._ragged_idx or query._ragged_idx != value._ragged_idx:
        raise ValueError(f'Expected query, key, and value to all be ragged on the same dimension, but got ragged dims {query._ragged_idx}, {key._ragged_idx}, and {value._ragged_idx}, respectively.')
    if attn_mask is not None:
        raise ValueError('Masks are not yet supported!')
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise ValueError(f'Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: {attn_mask.dtype}, and query.dtype: {query.dtype} instead.')