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
def BitVec(name, bv, ctx=None):
    """Return a bit-vector constant named `name`. `bv` may be the number of bits of a bit-vector sort.
    If `ctx=None`, then the global context is used.

    >>> x  = BitVec('x', 16)
    >>> is_bv(x)
    True
    >>> x.size()
    16
    >>> x.sort()
    BitVec(16)
    >>> word = BitVecSort(16)
    >>> x2 = BitVec('x', word)
    >>> eq(x, x2)
    True
    """
    if isinstance(bv, BitVecSortRef):
        ctx = bv.ctx
    else:
        ctx = _get_ctx(ctx)
        bv = BitVecSort(bv, ctx)
    return BitVecRef(Z3_mk_const(ctx.ref(), to_symbol(name, ctx), bv.ast), ctx)