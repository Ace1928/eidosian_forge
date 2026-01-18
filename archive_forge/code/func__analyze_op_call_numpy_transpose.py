import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_transpose(self, scope, equiv_set, loc, args, kws):
    in_arr = args[0]
    typ = self.typemap[in_arr.name]
    assert isinstance(typ, types.ArrayCompatible), 'Invalid np.transpose argument'
    shape = equiv_set._get_shape(in_arr)
    if len(args) == 1:
        return ArrayAnalysis.AnalyzeResult(shape=tuple(reversed(shape)))
    axes = [guard(find_const, self.func_ir, a) for a in args[1:]]
    if isinstance(axes[0], tuple):
        axes = list(axes[0])
    if None in axes:
        return None
    ret = [shape[i] for i in axes]
    return ArrayAnalysis.AnalyzeResult(shape=tuple(ret))