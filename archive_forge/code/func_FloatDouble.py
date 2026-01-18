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
def FloatDouble(ctx=None):
    """Floating-point 64-bit (double) sort."""
    ctx = _get_ctx(ctx)
    return FPSortRef(Z3_mk_fpa_sort_double(ctx.ref()), ctx)