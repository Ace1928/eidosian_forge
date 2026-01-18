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
def FreshConst(sort, prefix='c'):
    """Create a fresh constant of a specified sort"""
    ctx = _get_ctx(sort.ctx)
    return _to_expr_ref(Z3_mk_fresh_const(ctx.ref(), prefix, sort.ast), ctx)