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
def FiniteDomainVal(val, sort, ctx=None):
    """Return a Z3 finite-domain value. If `ctx=None`, then the global context is used.

    >>> s = FiniteDomainSort('S', 256)
    >>> FiniteDomainVal(255, s)
    255
    >>> FiniteDomainVal('100', s)
    100
    """
    if z3_debug():
        _z3_assert(is_finite_domain_sort(sort), 'Expected finite-domain sort')
    ctx = sort.ctx
    return FiniteDomainNumRef(Z3_mk_numeral(ctx.ref(), _to_int_str(val), sort.ast), ctx)