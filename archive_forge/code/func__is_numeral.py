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
def _is_numeral(ctx, a):
    return Z3_is_numeral_ast(ctx.ref(), a)