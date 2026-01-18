import functools
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten
def _check_out_dims_is_int_or_int_tuple(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    if not isinstance(out_dims, tuple) or not all((isinstance(out_dim, int) for out_dim in out_dims)):
        raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be an int or a tuple of int representing where in the outputs the vmapped dimension should appear.')