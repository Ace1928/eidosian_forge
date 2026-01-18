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
def UGT(a, b):
    """Create the Z3 expression (unsigned) `other > self`.

    Use the operator > for signed greater than.

    >>> x, y = BitVecs('x y', 32)
    >>> UGT(x, y)
    UGT(x, y)
    >>> (x > y).sexpr()
    '(bvsgt x y)'
    >>> UGT(x, y).sexpr()
    '(bvugt x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BoolRef(Z3_mk_bvugt(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)