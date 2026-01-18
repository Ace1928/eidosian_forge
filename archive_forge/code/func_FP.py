from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def FP(name, fpsort, ctx=None):
    """Return a floating-point constant named `name`.
    `fpsort` is the floating-point sort.
    If `ctx=None`, then the global context is used.

    >>> x  = FP('x', FPSort(8, 24))
    >>> is_fp(x)
    True
    >>> x.ebits()
    8
    >>> x.sort()
    FPSort(8, 24)
    >>> word = FPSort(8, 24)
    >>> x2 = FP('x', word)
    >>> eq(x, x2)
    True
    """
    if isinstance(fpsort, FPSortRef) and ctx is None:
        ctx = fpsort.ctx
    else:
        ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_const(ctx.ref(), to_symbol(name, ctx), fpsort.ast), ctx)