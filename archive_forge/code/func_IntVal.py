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
def IntVal(val, ctx=None):
    """Return a Z3 integer value. If `ctx=None`, then the global context is used.

    >>> IntVal(1)
    1
    >>> IntVal("100")
    100
    """
    ctx = _get_ctx(ctx)
    return IntNumRef(Z3_mk_numeral(ctx.ref(), _to_int_str(val), IntSort(ctx).ast), ctx)