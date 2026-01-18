import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_unary(self, scope, equiv_set, expr, lhs):
    require(expr.fn in UNARY_MAP_OP)
    if self._isarray(expr.value.name) or expr.fn == operator.add:
        return ArrayAnalysis.AnalyzeResult(shape=expr.value)
    return None