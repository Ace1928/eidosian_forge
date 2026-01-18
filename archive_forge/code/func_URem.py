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
def URem(a, b):
    """Create the Z3 expression (unsigned) remainder `self % other`.

    Use the operator % for signed modulus, and SRem() for signed remainder.

    >>> x = BitVec('x', 32)
    >>> y = BitVec('y', 32)
    >>> URem(x, y)
    URem(x, y)
    >>> URem(x, y).sort()
    BitVec(32)
    >>> (x % y).sexpr()
    '(bvsmod x y)'
    >>> URem(x, y).sexpr()
    '(bvurem x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_bvurem(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)