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
def EmptySet(s):
    """Create the empty set
    >>> EmptySet(IntSort())
    K(Int, False)
    """
    ctx = s.ctx
    return ArrayRef(Z3_mk_empty_set(ctx.ref(), s.ast), ctx)