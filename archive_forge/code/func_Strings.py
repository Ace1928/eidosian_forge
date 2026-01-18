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
def Strings(names, ctx=None):
    """Return a tuple of String constants. """
    ctx = _get_ctx(ctx)
    if isinstance(names, str):
        names = names.split(' ')
    return [String(name, ctx) for name in names]