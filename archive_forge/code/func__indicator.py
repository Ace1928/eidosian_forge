from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _indicator(n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr):
    core.static_assert(idx < n_dims)
    core.static_assert(pos == 0 or pos == 1)
    y = core.arange(0, 2)
    if pos == 0:
        y = 1 - y
    for n in core.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = core.expand_dims(y, n)
    return y