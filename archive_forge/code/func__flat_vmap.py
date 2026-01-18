import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
@doesnt_support_saved_tensors_hooks
def _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs):
    vmap_level = _vmap_increment_nesting(batch_size, randomness)
    try:
        batched_inputs = _create_batched_inputs(flat_in_dims, flat_args, vmap_level, args_spec)
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)
    finally:
        _vmap_decrement_nesting()