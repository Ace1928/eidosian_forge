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
def ULE(a, b):
    """Create the Z3 expression (unsigned) `other <= self`.

    Use the operator <= for signed less than or equal to.

    >>> x, y = BitVecs('x y', 32)
    >>> ULE(x, y)
    ULE(x, y)
    >>> (x <= y).sexpr()
    '(bvsle x y)'
    >>> ULE(x, y).sexpr()
    '(bvule x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BoolRef(Z3_mk_bvule(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)