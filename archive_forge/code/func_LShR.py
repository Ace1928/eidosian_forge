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
def LShR(a, b):
    """Create the Z3 expression logical right shift.

    Use the operator >> for the arithmetical right shift.

    >>> x, y = BitVecs('x y', 32)
    >>> LShR(x, y)
    LShR(x, y)
    >>> (x >> y).sexpr()
    '(bvashr x y)'
    >>> LShR(x, y).sexpr()
    '(bvlshr x y)'
    >>> BitVecVal(4, 3)
    4
    >>> BitVecVal(4, 3).as_signed_long()
    -4
    >>> simplify(BitVecVal(4, 3) >> 1).as_signed_long()
    -2
    >>> simplify(BitVecVal(4, 3) >> 1)
    6
    >>> simplify(LShR(BitVecVal(4, 3), 1))
    2
    >>> simplify(BitVecVal(2, 3) >> 1)
    1
    >>> simplify(LShR(BitVecVal(2, 3), 1))
    1
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_bvlshr(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)