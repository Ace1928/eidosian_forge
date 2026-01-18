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
def _cat_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    shape = tensors[0].shape
    concat_length = 0
    for tensor_idx, tensor in enumerate(tensors):
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                torch._check(length == common_length, lambda: f'Sizes of tensors must match except in dimension {dim}. Expected {common_length} but got {length} for tensor number {tensor_idx} in the list')
    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = concat_length
    return TensorMeta(tensors[0], shape=new_shape, strides=utils.make_contiguous_strides_for(new_shape))