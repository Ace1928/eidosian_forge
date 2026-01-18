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
def UDiv(a, b):
    """Create the Z3 expression (unsigned) division `self / other`.

    Use the operator / for signed division.

    >>> x = BitVec('x', 32)
    >>> y = BitVec('y', 32)
    >>> UDiv(x, y)
    UDiv(x, y)
    >>> UDiv(x, y).sort()
    BitVec(32)
    >>> (x / y).sexpr()
    '(bvsdiv x y)'
    >>> UDiv(x, y).sexpr()
    '(bvudiv x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_bvudiv(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)