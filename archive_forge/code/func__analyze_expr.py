import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_expr(self, scope, equiv_set, expr, lhs):
    fname = '_analyze_op_{}'.format(expr.op)
    try:
        fn = getattr(self, fname)
    except AttributeError:
        return None
    return guard(fn, scope, equiv_set, expr, lhs)