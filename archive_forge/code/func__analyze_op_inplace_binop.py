import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_inplace_binop(self, scope, equiv_set, expr, lhs):
    require(expr.fn in INPLACE_BINARY_MAP_OP)
    return self._analyze_broadcast(scope, equiv_set, expr.loc, [expr.lhs, expr.rhs], expr.fn)