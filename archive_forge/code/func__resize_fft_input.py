import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _resize_fft_input(x: TensorLikeType, dims: Tuple[int, ...], sizes: Tuple[int, ...]) -> TensorLikeType:
    """
    Fixes the shape of x such that x.size(dims[i]) == sizes[i],
    either by zero-padding, or by slicing x starting from 0.
    """
    assert len(dims) == len(sizes)
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue
        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_idx = len(pad_amount) - 2 * dims[i] - 1
            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]
        if x_sizes[dims[i]] > sizes[i]:
            x = x.narrow(dims[i], 0, sizes[i])
    return torch.constant_pad_nd(x, pad_amount) if must_copy else x