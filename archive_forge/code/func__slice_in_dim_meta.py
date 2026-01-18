import contextlib
import itertools
import operator
import weakref
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor, TypedStorage
from torch._C import _get_default_device
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
def _slice_in_dim_meta(a: TensorLikeType, start_index: int, limit_index: int, stride: int=1, axis: int=0) -> TensorLikeType:
    if axis < 0:
        msg = f'slice_in_dim: received a negative axis {axis}'
        raise ValueError(msg)
    if axis >= a.ndim:
        msg = f'slice_in_dim: axis {axis} is greater or equal to the rank {a.ndim} of the tensor'
        raise ValueError(msg)
    if start_index < 0:
        msg = f'slice_in_dim: received a negative start_index {start_index}'
        raise ValueError(msg)
    if start_index > a.shape[axis]:
        msg = f'slice_in_dim: start_index is greater than the length {start_index} of dimension {axis}'
        raise ValueError(msg)
    if limit_index > a.shape[axis]:
        msg = f'slice_in_dim: limit_index is greater than the length {limit_index} of dimension {axis}'
        raise ValueError(msg)
    if limit_index < start_index:
        msg = f'slice_in_dim: received a limit_index {limit_index} less than the start_index {start_index}'
        raise ValueError(msg)
    if stride < 0:
        msg = f'slice_in_dim: received a non-positive stride of {stride}!'
        raise ValueError(msg)
    start_indices = [0] * a.ndim
    limit_indices = list(a.shape)
    strides = [1] * a.ndim
    start_indices[axis] = start_index
    limit_indices[axis] = limit_index
    strides[axis] = stride
    return _slice_meta(a, start_indices, limit_indices, strides)