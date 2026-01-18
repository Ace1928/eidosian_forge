import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_getitem(self, scope, equiv_set, expr, lhs):
    result = self._index_to_shape(scope, equiv_set, expr.value, expr.index)
    if result[0] is not None:
        expr.index = result[0]
    return result[1]