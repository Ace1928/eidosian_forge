import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _reshape_tensor_or_tuple(u, shape):
    if isinstance(u, tuple):
        if not _is_sparse_any_tensor(u[0]):
            return (u[0].reshape(shape), u[1].reshape(shape))
    elif not _is_sparse_any_tensor(u):
        return u.reshape(shape)
    return u