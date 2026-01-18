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
def _find_iter_range(func_ir, range_iter_var, swapped):
    """Find the iterator's actual range if it is either range(n), or
    range(m, n), otherwise return raise GuardException.
    """
    debug_print = _make_debug_print('find_iter_range')
    range_iter_def = get_definition(func_ir, range_iter_var)
    debug_print('range_iter_var = ', range_iter_var, ' def = ', range_iter_def)
    require(isinstance(range_iter_def, ir.Expr) and range_iter_def.op == 'getiter')
    range_var = range_iter_def.value
    range_def = get_definition(func_ir, range_var)
    debug_print('range_var = ', range_var, ' range_def = ', range_def)
    require(isinstance(range_def, ir.Expr) and range_def.op == 'call')
    func_var = range_def.func
    func_def = get_definition(func_ir, func_var)
    debug_print('func_var = ', func_var, ' func_def = ', func_def)
    require(isinstance(func_def, ir.Global) and (func_def.value == range or func_def.value == numba.misc.special.prange))
    nargs = len(range_def.args)
    swapping = [('"array comprehension"', 'closure of'), range_def.func.loc]
    if nargs == 1:
        swapped[range_def.func.name] = swapping
        stop = get_definition(func_ir, range_def.args[0], lhs_only=True)
        return (0, range_def.args[0], func_def)
    elif nargs == 2:
        swapped[range_def.func.name] = swapping
        start = get_definition(func_ir, range_def.args[0], lhs_only=True)
        stop = get_definition(func_ir, range_def.args[1], lhs_only=True)
        return (start, stop, func_def)
    else:
        raise GuardException