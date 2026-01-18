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
def _to_flat_dense_if_sparse(tensor):
    if _is_sparse_any_tensor(tensor):
        return tensor.to_dense().reshape(-1)
    else:
        return tensor