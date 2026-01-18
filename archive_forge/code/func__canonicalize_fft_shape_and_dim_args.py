import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _canonicalize_fft_shape_and_dim_args(input: TensorLikeType, shape: Optional[ShapeType], dim: Optional[DimsType]) -> _ShapeAndDims:
    """Convert the shape and dim arguments into a canonical form where neither are optional"""
    input_dim = input.ndim
    input_sizes = input.shape
    if dim is not None:
        if not isinstance(dim, Sequence):
            dim = (dim,)
        ret_dims = utils.canonicalize_dims(input_dim, dim, wrap_scalar=False)
        torch._check(len(set(ret_dims)) == len(ret_dims), lambda: 'FFT dims must be unique')
    if shape is not None:
        if not isinstance(shape, Sequence):
            shape = (shape,)
        torch._check(dim is None or len(dim) == len(shape), lambda: 'When given, dim and shape arguments must have the same length')
        transform_ndim = len(shape)
        torch._check(transform_ndim <= input_dim, lambda: f'Got shape with {transform_ndim} values but input tensor only has {input_dim} dimensions.')
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))
        ret_shape = tuple((s if s != -1 else input_sizes[d] for s, d in zip(shape, ret_dims)))
    elif dim is None:
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        ret_shape = tuple((input_sizes[d] for d in ret_dims))
    for n in ret_shape:
        torch._check(n > 0, lambda: f'Invalid number of data points ({n}) specified')
    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)