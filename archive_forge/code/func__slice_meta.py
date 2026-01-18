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
def _slice_meta(a: TensorLikeType, start_indices: DimsSequenceType, limit_indices: DimsSequenceType, strides: Optional[StrideType]=None) -> TensorLikeType:
    _strides = strides if strides is not None else [1] * len(start_indices)
    if a.ndim != len(start_indices):
        msg = f'Attempting to slice tensor of rank {a.ndim} with start_indices of length {len(start_indices)}!'
        raise ValueError(msg)
    if a.ndim != len(limit_indices):
        msg = f'Attempting to slice tensor of rank {a.ndim} with limit_indices of length {len(limit_indices)}!'
        raise ValueError(msg)
    if a.ndim != len(_strides):
        msg = f'Attempting to slice tensor of rank {a.ndim} with strides of length {len(limit_indices)}!'
        raise ValueError(msg)
    for x, y in zip(start_indices, a.shape):
        if x < 0:
            msg = f'Attempting to slice a tensor with a negative start index of {x}!'
            raise ValueError(msg)
        if x > y:
            msg = f'Attempting to slice a tensor but a start index in {start_indices} is greater than the length of its corresponding dimension in shape {a.shape}'
            raise ValueError(msg)
    for x, y, z in zip(limit_indices, a.shape, start_indices):
        if x < 0:
            msg = f'Attempting to slice a tensor with a negative stop index of {x}!'
            raise ValueError(msg)
        if x > y:
            msg = f'Attempting to slice a tensor but a stop index in {limit_indices} is greater than the length of  its corresponding dimension in shape {a.shape}'
            raise ValueError(msg)
        if x < z:
            msg = f'Attempting to slice a tensor but a start index in {x} is greater than  its corresponding stop index {z}'
    for x in _strides:
        if x <= 0:
            msg = f'Attempting to slice a tensor with a non-positive step of {x}!'
            raise ValueError(msg)
    new_shape = []
    for x, y, z in zip(start_indices, limit_indices, _strides):
        new_shape.append(1 + (y - x - 1) // z)
    new_strides = []
    for x, y in zip(a.stride(), _strides):
        new_strides.append(x * y)
    return a.as_strided(new_shape, new_strides, a.storage_offset())