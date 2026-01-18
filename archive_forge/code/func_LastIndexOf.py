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
def LastIndexOf(s, substr):
    """Retrieve the last index of substring within a string"""
    ctx = None
    ctx = _get_ctx2(s, substr, ctx)
    s = _coerce_seq(s, ctx)
    substr = _coerce_seq(substr, ctx)
    return ArithRef(Z3_mk_seq_last_index(s.ctx_ref(), s.as_ast(), substr.as_ast()), s.ctx)