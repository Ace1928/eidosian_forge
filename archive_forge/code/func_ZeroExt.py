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
def ZeroExt(n, a):
    """Return a bit-vector expression with `n` extra zero-bits.

    >>> x = BitVec('x', 16)
    >>> n = ZeroExt(8, x)
    >>> n.size()
    24
    >>> n
    ZeroExt(8, x)
    >>> n.sort()
    BitVec(24)
    >>> v0 = BitVecVal(2, 2)
    >>> v0
    2
    >>> v0.size()
    2
    >>> v  = simplify(ZeroExt(6, v0))
    >>> v
    2
    >>> v.size()
    8
    """
    if z3_debug():
        _z3_assert(_is_int(n), 'First argument must be an integer')
        _z3_assert(is_bv(a), 'Second argument must be a Z3 bit-vector expression')
    return BitVecRef(Z3_mk_zero_ext(a.ctx_ref(), n, a.as_ast()), a.ctx)