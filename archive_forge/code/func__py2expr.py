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
def _py2expr(a, ctx=None):
    if isinstance(a, bool):
        return BoolVal(a, ctx)
    if _is_int(a):
        return IntVal(a, ctx)
    if isinstance(a, float):
        return RealVal(a, ctx)
    if isinstance(a, str):
        return StringVal(a, ctx)
    if is_expr(a):
        return a
    if z3_debug():
        _z3_assert(False, 'Python bool, int, long or float expected')