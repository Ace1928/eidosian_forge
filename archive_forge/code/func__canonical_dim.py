import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _canonical_dim(dim: DimOrDims, ndim: int) -> Tuple[int, ...]:
    """Return dim argument as a tuple of sorted dim values."""
    dims: List[int] = []
    if dim == ():
        dim = None
    if dim is None:
        return tuple(range(ndim))
    ndim = max(ndim, 1)
    dim_ = (dim,) if isinstance(dim, (int, torch.SymInt)) else dim
    for d in dim_:
        if d in dims:
            raise RuntimeError(f'dim={d} appears multiple times in the list of dims')
        if d >= ndim or d < -ndim:
            raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})')
        dims.append(d % ndim)
    return tuple(sorted(dims))