from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
def _check_bias_alignment(reasons: List[str], attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> None:
    attn_bias_tensor = _get_tensor_bias(attn_bias)
    if attn_bias_tensor is not None:
        alignment = 128 // torch.finfo(attn_bias_tensor.dtype).bits
        show_padding_hint = False
        for d in range(attn_bias_tensor.ndim - 1):
            if attn_bias_tensor.stride(d) % alignment != 0:
                reasons.append(f'attn_bias.stride(-2) % {alignment} != 0 (attn_bias.stride() = {attn_bias_tensor.stride()})')
                show_padding_hint = True
        if show_padding_hint:
            reasons.append('HINT: To use an `attn_bias` with a sequence length that is not a multiple of 8, you need to ensure memory is aligned by slicing a bigger tensor. Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]` instead of `torch.zeros([1, 1, 5, 5])`')
        if attn_bias_tensor.stride(-1) > 1:
            reasons.append(f'attn_bias.stride(-1) > 1 (attn_bias.stride() = {attn_bias_tensor.stride()}) - you should call `.contiguous()` on the bias')