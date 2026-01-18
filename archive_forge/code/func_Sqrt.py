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
def Sqrt(a, ctx=None):
    """ Return a Z3 expression which represents the square root of a.

    >>> x = Real('x')
    >>> Sqrt(x)
    x**(1/2)
    """
    if not is_expr(a):
        ctx = _get_ctx(ctx)
        a = RealVal(a, ctx)
    return a ** '1/2'