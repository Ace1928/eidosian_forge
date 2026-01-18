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
def Implies(a, b, ctx=None):
    """Create a Z3 implies expression.

    >>> p, q = Bools('p q')
    >>> Implies(p, q)
    Implies(p, q)
    """
    ctx = _get_ctx(_ctx_from_ast_arg_list([a, b], ctx))
    s = BoolSort(ctx)
    a = s.cast(a)
    b = s.cast(b)
    return BoolRef(Z3_mk_implies(ctx.ref(), a.as_ast(), b.as_ast()), ctx)