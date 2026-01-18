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
def _iter_tensors(x: Union[torch.Tensor, Iterable[torch.Tensor]], only_requiring_grad: bool=False) -> Iterable[torch.Tensor]:
    if is_tensor_like(x):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, collections.abc.Iterable) and (not isinstance(x, str)):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)