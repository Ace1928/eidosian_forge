import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_eye(self, scope, equiv_set, loc, args, kws):
    if len(args) > 0:
        N = args[0]
    elif 'N' in kws:
        N = kws['N']
    else:
        raise errors.UnsupportedRewriteError("Expect one argument (or 'N') to eye function", loc=loc)
    if 'M' in kws:
        M = kws['M']
    else:
        M = N
    return ArrayAnalysis.AnalyzeResult(shape=(N, M))