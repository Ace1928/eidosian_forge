import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_exhaust_iter(self, scope, equiv_set, expr, lhs):
    var = expr.value
    typ = self.typemap[var.name]
    if isinstance(typ, types.BaseTuple):
        require(len(typ) == expr.count)
        require(equiv_set.has_shape(var))
        return ArrayAnalysis.AnalyzeResult(shape=var)
    return None