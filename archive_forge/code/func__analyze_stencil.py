import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_stencil(self, scope, equiv_set, stencil_func, loc, args, kws):
    std_idx_arrs = stencil_func.options.get('standard_indexing', ())
    kernel_arg_names = stencil_func.kernel_ir.arg_names
    if isinstance(std_idx_arrs, str):
        std_idx_arrs = (std_idx_arrs,)
    rel_idx_arrs = []
    assert len(args) > 0 and len(args) == len(kernel_arg_names)
    for arg, var in zip(kernel_arg_names, args):
        typ = self.typemap[var.name]
        if isinstance(typ, types.ArrayCompatible) and (not arg in std_idx_arrs):
            rel_idx_arrs.append(var)
    n = len(rel_idx_arrs)
    require(n > 0)
    asserts = self._call_assert_equiv(scope, loc, equiv_set, rel_idx_arrs)
    shape = equiv_set.get_shape(rel_idx_arrs[0])
    return ArrayAnalysis.AnalyzeResult(shape=shape, pre=asserts)