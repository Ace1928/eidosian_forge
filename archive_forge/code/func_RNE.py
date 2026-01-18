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
def RNE(ctx=None):
    ctx = _get_ctx(ctx)
    return FPRMRef(Z3_mk_fpa_round_nearest_ties_to_even(ctx.ref()), ctx)