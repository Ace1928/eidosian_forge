from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
def _get_sort_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, 'Currently only support sorting on the last dimension'
    return core.constexpr(dim)