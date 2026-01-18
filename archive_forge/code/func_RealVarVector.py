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
def RealVarVector(n, ctx=None):
    """
    Create a list of Real free variables.
    The variables have ids: 0, 1, ..., n-1

    >>> x0, x1, x2, x3 = RealVarVector(4)
    >>> x2
    Var(2)
    """
    return [RealVar(i, ctx) for i in range(n)]