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
def BitVecSort(sz, ctx=None):
    """Return a Z3 bit-vector sort of the given size. If `ctx=None`, then the global context is used.

    >>> Byte = BitVecSort(8)
    >>> Word = BitVecSort(16)
    >>> Byte
    BitVec(8)
    >>> x = Const('x', Byte)
    >>> eq(x, BitVec('x', 8))
    True
    """
    ctx = _get_ctx(ctx)
    return BitVecSortRef(Z3_mk_bv_sort(ctx.ref(), sz), ctx)