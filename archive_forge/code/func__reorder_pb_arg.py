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
def _reorder_pb_arg(arg):
    a, b = arg
    if not _is_int(b) and _is_int(a):
        return (b, a)
    return arg