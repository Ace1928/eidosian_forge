import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_diag(self, scope, equiv_set, loc, args, kws):
    assert len(args) > 0
    a = args[0]
    assert isinstance(a, ir.Var)
    atyp = self.typemap[a.name]
    if isinstance(atyp, types.ArrayCompatible):
        if atyp.ndim == 2:
            if 'k' in kws:
                k = kws['k']
                if not equiv_set.is_equiv(k, 0):
                    return None
            m, n = equiv_set._get_shape(a)
            if equiv_set.is_equiv(m, n):
                return ArrayAnalysis.AnalyzeResult(shape=(m,))
        elif atyp.ndim == 1:
            m, = equiv_set._get_shape(a)
            return ArrayAnalysis.AnalyzeResult(shape=(m, m))
    return None