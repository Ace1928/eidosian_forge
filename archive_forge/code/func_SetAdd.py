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
def SetAdd(s, e):
    """ Add element e to set s
    >>> a = Const('a', SetSort(IntSort()))
    >>> SetAdd(a, 1)
    Store(a, 1, True)
    """
    ctx = _ctx_from_ast_arg_list([s, e])
    e = _py2expr(e, ctx)
    return ArrayRef(Z3_mk_set_add(ctx.ref(), s.as_ast(), e.as_ast()), ctx)