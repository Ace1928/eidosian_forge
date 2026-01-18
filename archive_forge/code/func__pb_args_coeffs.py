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
def _pb_args_coeffs(args, default_ctx=None):
    args = _get_args_ast_list(args)
    if len(args) == 0:
        return (_get_ctx(default_ctx), 0, (Ast * 0)(), (ctypes.c_int * 0)())
    args = [_reorder_pb_arg(arg) for arg in args]
    args, coeffs = zip(*args)
    if z3_debug():
        _z3_assert(len(args) > 0, 'Non empty list of arguments expected')
    ctx = _ctx_from_ast_arg_list(args)
    if z3_debug():
        _z3_assert(ctx is not None, 'At least one of the arguments must be a Z3 expression')
    args = _coerce_expr_list(args, ctx)
    _args, sz = _to_ast_array(args)
    _coeffs = (ctypes.c_int * len(coeffs))()
    for i in range(len(coeffs)):
        _z3_check_cint_overflow(coeffs[i], 'coefficient')
        _coeffs[i] = coeffs[i]
    return (ctx, sz, _args, _coeffs, args)