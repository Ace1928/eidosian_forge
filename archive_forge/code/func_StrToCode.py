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
def StrToCode(s):
    """Convert a unit length string to integer code"""
    if not is_expr(s):
        s = _py2expr(s)
    return ArithRef(Z3_mk_string_to_code(s.ctx_ref(), s.as_ast()), s.ctx)