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
def Const(name, sort):
    """Create a constant of the given sort.

    >>> Const('x', IntSort())
    x
    """
    if z3_debug():
        _z3_assert(isinstance(sort, SortRef), 'Z3 sort expected')
    ctx = sort.ctx
    return _to_expr_ref(Z3_mk_const(ctx.ref(), to_symbol(name, ctx), sort.ast), ctx)