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
def IntToStr(s):
    """Convert integer expression to string"""
    if not is_expr(s):
        s = _py2expr(s)
    return SeqRef(Z3_mk_int_to_str(s.ctx_ref(), s.as_ast()), s.ctx)