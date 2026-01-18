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
def _coerce_exprs(a, b, ctx=None):
    if not is_expr(a) and (not is_expr(b)):
        a = _py2expr(a, ctx)
        b = _py2expr(b, ctx)
    if isinstance(a, str) and isinstance(b, SeqRef):
        a = StringVal(a, b.ctx)
    if isinstance(b, str) and isinstance(a, SeqRef):
        b = StringVal(b, a.ctx)
    if isinstance(a, float) and isinstance(b, ArithRef):
        a = RealVal(a, b.ctx)
    if isinstance(b, float) and isinstance(a, ArithRef):
        b = RealVal(b, a.ctx)
    s = None
    s = _coerce_expr_merge(s, a)
    s = _coerce_expr_merge(s, b)
    a = s.cast(a)
    b = s.cast(b)
    return (a, b)