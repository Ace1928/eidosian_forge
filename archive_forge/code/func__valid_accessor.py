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
def _valid_accessor(acc):
    """Return `True` if acc is pair of the form (String, Datatype or Sort). """
    if not isinstance(acc, tuple):
        return False
    if len(acc) != 2:
        return False
    return isinstance(acc[0], str) and (isinstance(acc[1], Datatype) or is_sort(acc[1]))