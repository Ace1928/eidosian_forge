import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _maybe_convert_to_dtype(a, dtype):
    if isinstance(a, TensorLike):
        if a.dtype != dtype:
            return a.to(dtype)
        return a
    if isinstance(a, Number):
        return utils.dtype_to_type_ctor(dtype)(a)
    if isinstance(a, Sequence):
        return tuple((_maybe_convert_to_dtype(x, dtype) for x in a))
    if a is None:
        return None
    raise ValueError(f'Received type {type(a)} that is neither a tensor or a number!')