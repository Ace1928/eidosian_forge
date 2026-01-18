import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def _mk_range_args(typemap, start, stop, step, scope, loc):
    nodes = []
    if isinstance(stop, ir.Var):
        g_stop_var = stop
    else:
        assert isinstance(stop, int)
        g_stop_var = ir.Var(scope, mk_unique_var('$range_stop'), loc)
        if typemap:
            typemap[g_stop_var.name] = types.intp
        stop_assign = ir.Assign(ir.Const(stop, loc), g_stop_var, loc)
        nodes.append(stop_assign)
    if start == 0 and step == 1:
        return (nodes, [g_stop_var])
    if isinstance(start, ir.Var):
        g_start_var = start
    else:
        assert isinstance(start, int)
        g_start_var = ir.Var(scope, mk_unique_var('$range_start'), loc)
        if typemap:
            typemap[g_start_var.name] = types.intp
        start_assign = ir.Assign(ir.Const(start, loc), g_start_var, loc)
        nodes.append(start_assign)
    if step == 1:
        return (nodes, [g_start_var, g_stop_var])
    if isinstance(step, ir.Var):
        g_step_var = step
    else:
        assert isinstance(step, int)
        g_step_var = ir.Var(scope, mk_unique_var('$range_step'), loc)
        if typemap:
            typemap[g_step_var.name] = types.intp
        step_assign = ir.Assign(ir.Const(step, loc), g_step_var, loc)
        nodes.append(step_assign)
    return (nodes, [g_start_var, g_stop_var, g_step_var])