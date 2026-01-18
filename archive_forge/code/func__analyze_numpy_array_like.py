import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_numpy_array_like(self, scope, equiv_set, args, kws):
    assert len(args) > 0
    var = args[0]
    typ = self.typemap[var.name]
    if isinstance(typ, types.Integer):
        return ArrayAnalysis.AnalyzeResult(shape=(1,))
    elif isinstance(typ, types.ArrayCompatible) and equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var)
    return None