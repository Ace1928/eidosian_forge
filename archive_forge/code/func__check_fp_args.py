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
def _check_fp_args(a, b):
    if z3_debug():
        _z3_assert(is_fp(a) or is_fp(b), 'First or second argument must be a Z3 floating-point expression')