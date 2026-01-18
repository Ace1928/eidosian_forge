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
def _get_ctx2(a, b, ctx=None):
    if is_expr(a):
        return a.ctx
    if is_expr(b):
        return b.ctx
    if ctx is None:
        ctx = main_ctx()
    return ctx