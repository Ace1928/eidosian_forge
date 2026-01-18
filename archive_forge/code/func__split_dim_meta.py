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
def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)
    inner_length = a.shape[dim] // outer_length
    if a.shape[dim] % outer_length != 0:
        msg = 'Attempting to split dimension of length {}, but outer length of {} divides it with a remainder!'.format(a.shape[dim], outer_length)
        raise ValueError(msg)
    new_shape: List[int] = []
    new_strides: List[int] = []
    for idx in range(a.ndim):
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])
    return a.as_strided(new_shape, new_strides, a.storage_offset())