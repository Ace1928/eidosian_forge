import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _fftn_c2c(function_name: str, input: TensorLikeType, shape: Tuple[int, ...], dim: Tuple[int, ...], norm: NormType, forward: bool) -> TensorLikeType:
    """Common code for n-dimensional complex to complex FFTs (fftn or ifftn)"""
    torch._check(input.dtype.is_complex, lambda: f'{function_name} expects a complex input tensor, but got {input.dtype}')
    x = _resize_fft_input(input, dim, shape)
    output = prims.fft_c2c(x, dim=dim, forward=forward)
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)