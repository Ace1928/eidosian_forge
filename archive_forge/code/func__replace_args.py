from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(f'new_args should be of size 1, was of size {len(new_args)}')
        return tuple((new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))))
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(f'new_args should have the same size as argnums. Argnums size {len(argnums)}, new_args size {len(new_args)}')

        def get_right_elem(i):
            return new_args[argnums.index(i)] if i in argnums else old_args[i]
        return tuple((get_right_elem(i) for i in range(len(old_args))))
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')