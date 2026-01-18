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
def _inline_closure(self, work_list, block, i, func_def):
    require(isinstance(func_def, ir.Expr) and func_def.op == 'make_function')
    inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, func_def, work_list=work_list, callee_validator=callee_ir_validator)
    return True