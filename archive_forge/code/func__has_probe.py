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
def _has_probe(args):
    """Return `True` if one of the elements of the given collection is a Z3 probe."""
    for arg in args:
        if is_probe(arg):
            return True
    return False