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
def ParAndThen(t1, t2, ctx=None):
    """Alias for ParThen(t1, t2, ctx)."""
    return ParThen(t1, t2, ctx)