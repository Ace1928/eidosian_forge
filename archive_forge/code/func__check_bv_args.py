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
def _check_bv_args(a, b):
    if z3_debug():
        _z3_assert(is_bv(a) or is_bv(b), 'First or second argument must be a Z3 bit-vector expression')