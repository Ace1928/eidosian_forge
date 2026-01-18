import functools
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten
def _vmap(func: Callable, in_dims: in_dims_t=0, out_dims: out_dims_t=0, allow_none_pass_through: bool=False) -> Callable:

    @functools.wraps(func)
    def wrapped(*args):
        _check_out_dims_is_int_or_int_tuple(out_dims, func)
        vmap_level = torch._C._vmapmode_increment_nesting()
        try:
            batched_inputs, batch_size = _create_batched_inputs(in_dims, args, vmap_level, func)
            batched_outputs = func(*batched_inputs)
            if not allow_none_pass_through:
                _validate_outputs(batched_outputs, func)
            return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func, allow_none_pass_through=allow_none_pass_through)
        finally:
            torch._C._vmapmode_decrement_nesting()
    return wrapped