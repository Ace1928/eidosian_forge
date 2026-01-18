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
def FreshBool(prefix='b', ctx=None):
    """Return a fresh Boolean constant in the given context using the given prefix.

    If `ctx=None`, then the global context is used.

    >>> b1 = FreshBool()
    >>> b2 = FreshBool()
    >>> eq(b1, b2)
    False
    """
    ctx = _get_ctx(ctx)
    return BoolRef(Z3_mk_fresh_const(ctx.ref(), prefix, BoolSort(ctx).ast), ctx)