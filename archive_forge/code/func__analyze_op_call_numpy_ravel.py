import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_ravel(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1
    var = args[0]
    typ = self.typemap[var.name]
    assert isinstance(typ, types.ArrayCompatible)
    if typ.ndim == 1 and equiv_set.has_shape(var):
        if typ.layout == 'C':
            return ArrayAnalysis.AnalyzeResult(shape=var, rhs=var)
        else:
            return ArrayAnalysis.AnalyzeResult(shape=var)
    return None