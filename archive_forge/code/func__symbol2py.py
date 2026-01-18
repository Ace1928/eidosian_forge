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
def _symbol2py(ctx, s):
    """Convert a Z3 symbol back into a Python object. """
    if Z3_get_symbol_kind(ctx.ref(), s) == Z3_INT_SYMBOL:
        return 'k!%s' % Z3_get_symbol_int(ctx.ref(), s)
    else:
        return Z3_get_symbol_string(ctx.ref(), s)