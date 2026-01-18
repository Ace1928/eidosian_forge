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
def Product(*args):
    """Create the product of the Z3 expressions.

    >>> a, b, c = Ints('a b c')
    >>> Product(a, b, c)
    a*b*c
    >>> Product([a, b, c])
    a*b*c
    >>> A = IntVector('a', 5)
    >>> Product(A)
    a__0*a__1*a__2*a__3*a__4
    """
    args = _get_args(args)
    if len(args) == 0:
        return 1
    ctx = _ctx_from_ast_arg_list(args)
    if ctx is None:
        return _reduce(lambda a, b: a * b, args, 1)
    args = _coerce_expr_list(args, ctx)
    if is_bv(args[0]):
        return _reduce(lambda a, b: a * b, args, 1)
    else:
        _args, sz = _to_ast_array(args)
        return ArithRef(Z3_mk_mul(ctx.ref(), sz, _args), ctx)