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
def _maybe_wrap_functional_tensor(maybe_tensor, level, *, _python_functionalize: bool=False):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    _assert_wrapped_functional(maybe_tensor, wrapped)
    if _python_functionalize:
        out = FunctionalTensor(wrapped)
        torch._mirror_autograd_meta_to(maybe_tensor, out)
        return out
    return wrapped