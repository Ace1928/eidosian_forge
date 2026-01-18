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
def _mk_fp_unary_pred(f, a, ctx):
    ctx = _get_ctx(ctx)
    [a] = _coerce_fp_expr_list([a], ctx)
    if z3_debug():
        _z3_assert(is_fp(a), 'First argument must be a Z3 floating-point expression')
    return BoolRef(f(ctx.ref(), a.as_ast()), ctx)