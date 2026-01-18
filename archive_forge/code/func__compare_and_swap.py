from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _compare_and_swap(x, desc_mask, n_dims: core.constexpr, idx: core.constexpr):
    l = _take_slice(x, n_dims, idx, 0)
    r = _take_slice(x, n_dims, idx, 1)
    x_int = x
    l_int = l
    r_int = r
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError('Unsupported dtype')
        x_int = x.to(dtype_int, bitcast=True)
        l_int = l.to(dtype_int, bitcast=True)
        r_int = r.to(dtype_int, bitcast=True)
    desc_mask = desc_mask.to(x_int.dtype)
    zero = zeros_like(x_int)
    y = x_int ^ core.where((l > r) ^ desc_mask, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)
    return y