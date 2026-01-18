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
def BVAddNoUnderflow(a, b):
    """A predicate the determines that signed bit-vector addition does not underflow"""
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BoolRef(Z3_mk_bvadd_no_underflow(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)