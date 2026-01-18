from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def _to_numeral(num, ctx=None):
    if isinstance(num, Numeral):
        return num
    else:
        return Numeral(num, ctx)