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
def CharVal(ch, ctx=None):
    ctx = _get_ctx(ctx)
    if isinstance(ch, str):
        ch = ord(ch)
    if not isinstance(ch, int):
        raise Z3Exception('character value should be an ordinal')
    return _to_expr_ref(Z3_mk_char(ctx.ref(), ch), ctx)