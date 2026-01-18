import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def _wrap_jagged_dims(ndim, dims, op_name):
    from torch._prims_common import canonicalize_dims
    wrapped_dims = [canonicalize_dims(ndim, d) for d in dims]
    zero_in_dims = 0 in wrapped_dims
    one_in_dims = 1 in wrapped_dims
    if zero_in_dims ^ one_in_dims:
        apply, not_apply = ('batch', 'ragged') if zero_in_dims else ('ragged', 'batch')
        raise RuntimeError(f'{op_name}(): applying over the {apply} dimension, but not the {not_apply} dimension is not supported for NestedTensor')
    return (tuple((_outer_to_inner_dim(ndim, d) for d in dims if d != 0)), zero_in_dims)