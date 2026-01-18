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
def _safe_copy_out(*, copy_from: TensorLikeType, copy_to: TensorLikeType, exact_dtype: bool=False):
    if copy_from.device != copy_to.device:
        msg = 'Attempting to copy from device {} to device {}, but cross-device copies are not allowed!'.format(copy_from.device, copy_to.device)
        raise RuntimeError(msg)
    if exact_dtype:
        torch._check(copy_from.dtype == copy_to.dtype, lambda: f'Expected out tensor to have dtype {copy_from.dtype} but got {copy_to.dtype} instead')
    else:
        torch._check(utils.can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype), lambda: f"Attempting to cast from {copy_from.dtype} to out tensor with dtype {copy_to.dtype}, but this can't be cast because it is not safe!")
    return copy_to.copy_(copy_from)