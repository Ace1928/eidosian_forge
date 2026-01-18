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
def PbEq(args, k, ctx=None):
    """Create a Pseudo-Boolean equality k constraint.

    >>> a, b, c = Bools('a b c')
    >>> f = PbEq(((a,1),(b,3),(c,2)), 3)
    """
    _z3_check_cint_overflow(k, 'k')
    ctx, sz, _args, _coeffs, args = _pb_args_coeffs(args)
    return BoolRef(Z3_mk_pbeq(ctx.ref(), sz, _args, _coeffs, k), ctx)