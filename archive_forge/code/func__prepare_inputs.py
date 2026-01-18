from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple
import torch
from ... import _is_triton_available
from ..common import register_operator
from .attn_bias import LowerTriangularMask
from .common import (
def _prepare_inputs(inp: Inputs) -> Inputs:
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, torch.Tensor) and attn_bias.ndim == 3:
        B = inp.query.shape[0]
        h = attn_bias.shape[0] // B
        attn_bias = attn_bias.reshape(B, h, attn_bias.shape[1], attn_bias.shape[2])
    query, key, value = [x if x.stride(-1) == 1 else x.contiguous() for x in [inp.query, inp.key, inp.value]]
    return replace(inp, attn_bias=attn_bias, query=query, key=key, value=value)