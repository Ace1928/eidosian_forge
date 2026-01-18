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
def ULT(a, b):
    """Create the Z3 expression (unsigned) `other < self`.

    Use the operator < for signed less than.

    >>> x, y = BitVecs('x y', 32)
    >>> ULT(x, y)
    ULT(x, y)
    >>> (x < y).sexpr()
    '(bvslt x y)'
    >>> ULT(x, y).sexpr()
    '(bvult x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BoolRef(Z3_mk_bvult(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)