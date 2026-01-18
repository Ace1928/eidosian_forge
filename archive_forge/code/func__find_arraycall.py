import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def _find_arraycall(func_ir, block):
    """Look for statement like "x = numpy.array(y)" or "x[..] = y"
    immediately after the closure call that creates list y (the i-th
    statement in block).  Return the statement index if found, or
    raise GuardException.
    """
    array_var = None
    list_var_dead_after_array_call = False
    list_var = None
    i = 0
    while i < len(block.body):
        instr = block.body[i]
        if isinstance(instr, ir.Del):
            if list_var and array_var and (instr.value == list_var.name):
                list_var_dead_after_array_call = True
                break
            pass
        elif isinstance(instr, ir.Assign):
            lhs = instr.target
            expr = instr.value
            if guard(find_callname, func_ir, expr) == ('array', 'numpy') and isinstance(expr.args[0], ir.Var):
                list_var = expr.args[0]
                array_var = lhs
                array_stmt_index = i
                array_kws = dict(expr.kws)
        elif isinstance(instr, ir.SetItem) and isinstance(instr.value, ir.Var) and (not list_var):
            list_var = instr.value
            array_var = instr.target
            array_def = get_definition(func_ir, array_var)
            require(guard(_find_unsafe_empty_inferred, func_ir, array_def))
            array_stmt_index = i
            array_kws = {}
        else:
            break
        i = i + 1
    require(array_var and list_var_dead_after_array_call)
    _make_debug_print('find_array_call')(block.body[array_stmt_index])
    return (list_var, array_stmt_index, array_kws)