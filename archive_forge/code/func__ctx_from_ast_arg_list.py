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
def _ctx_from_ast_arg_list(args, default_ctx=None):
    ctx = None
    for a in args:
        if is_ast(a) or is_probe(a):
            if ctx is None:
                ctx = a.ctx
            elif z3_debug():
                _z3_assert(ctx == a.ctx, 'Context mismatch')
    if ctx is None:
        ctx = default_ctx
    return ctx