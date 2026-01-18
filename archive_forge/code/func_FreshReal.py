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
def FreshReal(prefix='b', ctx=None):
    """Return a fresh real constant in the given context using the given prefix.

    >>> x = FreshReal()
    >>> y = FreshReal()
    >>> eq(x, y)
    False
    >>> x.sort()
    Real
    """
    ctx = _get_ctx(ctx)
    return ArithRef(Z3_mk_fresh_const(ctx.ref(), prefix, RealSort(ctx).ast), ctx)