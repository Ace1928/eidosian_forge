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
def Distinct(*args):
    """Create a Z3 distinct expression.

    >>> x = Int('x')
    >>> y = Int('y')
    >>> Distinct(x, y)
    x != y
    >>> z = Int('z')
    >>> Distinct(x, y, z)
    Distinct(x, y, z)
    >>> simplify(Distinct(x, y, z))
    Distinct(x, y, z)
    >>> simplify(Distinct(x, y, z), blast_distinct=True)
    And(Not(x == y), Not(x == z), Not(y == z))
    """
    args = _get_args(args)
    ctx = _ctx_from_ast_arg_list(args)
    if z3_debug():
        _z3_assert(ctx is not None, 'At least one of the arguments must be a Z3 expression')
    args = _coerce_expr_list(args, ctx)
    _args, sz = _to_ast_array(args)
    return BoolRef(Z3_mk_distinct(ctx.ref(), sz, _args), ctx)