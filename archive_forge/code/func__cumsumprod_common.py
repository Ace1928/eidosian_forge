import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _cumsumprod_common(func, init, a: TensorLikeType, dim: int, *, dtype: Optional[torch.dtype]=None, out: Optional[Tensor]=None) -> TensorLikeType:
    ndim = a.ndim
    dim = utils.canonicalize_dim(ndim, dim)
    if ndim == 0:
        return func(a.unsqueeze(0), dim=0, dtype=dtype, out=out)
    a = a.unsqueeze(dim + 1)
    rg = torch.arange(a.shape[dim], device=a.device)
    mask = rg.unsqueeze(1) <= rg
    for _ in range(ndim - dim - 1):
        mask = mask.unsqueeze(-1)
    masked_a = torch.where(mask, a, init)
    return func(masked_a, dim=dim, dtype=dtype, out=out)