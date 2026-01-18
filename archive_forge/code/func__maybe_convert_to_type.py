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
def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType:
    if not isinstance(a, Number):
        msg = f'Found unknown type {type(a)} when trying to convert scalars!'
        raise ValueError(msg)
    if not utils.is_weakly_lesser_type(type(a), typ):
        msg = f'Scalar {a} of type {type(a)} cannot be safely cast to type {typ}!'
        raise ValueError(msg)
    return typ(a)