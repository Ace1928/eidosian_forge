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
def _mk_fp_bin_norm(f, a, b, ctx):
    ctx = _get_ctx(ctx)
    [a, b] = _coerce_fp_expr_list([a, b], ctx)
    if z3_debug():
        _z3_assert(is_fp(a) or is_fp(b), 'First or second argument must be a Z3 floating-point expression')
    return FPRef(f(ctx.ref(), a.as_ast(), b.as_ast()), ctx)