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
def TensorMeta(tensorlike: Optional[Union[NumberType, torch.Tensor]]=None, *, shape: Optional[ShapeType]=None, strides: Optional[StrideType]=None, dtype: Optional[torch.dtype]=None, device: Optional[Union[torch.device, str]]=None):
    if isinstance(tensorlike, Number):
        assert not shape and (shape is None or isinstance(shape, Sequence))
        assert not strides and (strides is None or isinstance(strides, Sequence))
        inferred_shape: Tuple[int, ...] = ()
        inferred_strides: Tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = torch.device('cpu')
    elif tensorlike is not None:
        assert isinstance(tensorlike, torch.Tensor)
        inferred_shape = tuple(tensorlike.shape)
        inferred_strides = tuple(tensorlike.stride())
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        assert shape is not None
        assert strides is not None
        assert dtype is not None
        assert device is not None
    shape = inferred_shape if shape is None else tuple(shape)
    strides = inferred_strides if strides is None else tuple(strides)
    dtype = inferred_dtype if dtype is None else dtype
    device = inferred_device if device is None else device
    if isinstance(device, str):
        device = torch.device(device)
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)