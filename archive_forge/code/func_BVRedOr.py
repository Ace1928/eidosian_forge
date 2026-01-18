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
def BVRedOr(a):
    """Return the reduction-or expression of `a`."""
    if z3_debug():
        _z3_assert(is_bv(a), 'First argument must be a Z3 bit-vector expression')
    return BitVecRef(Z3_mk_bvredor(a.ctx_ref(), a.as_ast()), a.ctx)