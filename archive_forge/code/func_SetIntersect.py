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
def SetIntersect(*args):
    """ Take the union of sets
    >>> a = Const('a', SetSort(IntSort()))
    >>> b = Const('b', SetSort(IntSort()))
    >>> SetIntersect(a, b)
    intersection(a, b)
    """
    args = _get_args(args)
    ctx = _ctx_from_ast_arg_list(args)
    _args, sz = _to_ast_array(args)
    return ArrayRef(Z3_mk_set_intersect(ctx.ref(), sz, _args), ctx)