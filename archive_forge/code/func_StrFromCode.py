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
def StrFromCode(c):
    """Convert code to a string"""
    if not is_expr(c):
        c = _py2expr(c)
    return SeqRef(Z3_mk_string_from_code(c.ctx_ref(), c.as_ast()), c.ctx)