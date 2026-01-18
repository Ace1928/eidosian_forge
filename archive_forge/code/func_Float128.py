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
def Float128(ctx=None):
    """Floating-point 128-bit (quadruple) sort."""
    ctx = _get_ctx(ctx)
    return FPSortRef(Z3_mk_fpa_sort_128(ctx.ref()), ctx)