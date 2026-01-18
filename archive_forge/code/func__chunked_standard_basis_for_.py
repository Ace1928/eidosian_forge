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
def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    assert chunk_size is None or chunk_size > 0
    total_numel = sum(tensor_numels)
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:
        chunk_size = total_numel
        chunk_numels = [total_numel]
    diag_start_indices = (0, *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind())
    for chunk_idx, total_numel in enumerate(chunk_numels):
        chunks = tuple((tensor.new_zeros(total_numel, tensor_numel) for tensor, tensor_numel in zip(tensors, tensor_numels)))
        for chunk, diag_start_idx in zip(chunks, diag_start_indices):
            chunk.diagonal(diag_start_idx + chunk_idx * chunk_size).fill_(1)
        chunks = tuple((chunk.view(total_numel, *tensor.shape) for chunk, tensor in zip(chunks, tensors)))
        yield chunks