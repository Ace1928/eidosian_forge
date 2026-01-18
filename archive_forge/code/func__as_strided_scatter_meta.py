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
def _as_strided_scatter_meta(input: TensorLikeType, src: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int) -> TensorLikeType:
    utils.validate_shape(size)
    utils.validate_strides(stride)
    required_size = utils.compute_required_storage_length(size, stride, storage_offset)
    torch._check(input.numel() >= required_size, lambda: f'as_strided_scatter: sizes {size}, strides {stride}, storage offset {storage_offset}  and itemsize {input.element_size()} requiring a storage size of {required_size * input.element_size()} are out of bounds for storage of size {input.numel() * input.element_size()}')
    torch._check(utils.is_same_shape(src.shape, size), lambda: f'expected src to have a size equal to the slice of self. src size = {src.shape}, slice size = {size}')
    return utils.clone_preserve_strides(input)