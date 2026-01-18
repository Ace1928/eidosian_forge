import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_full_like(self, scope, equiv_set, loc, args, kws):
    return self._analyze_numpy_array_like(scope, equiv_set, args, kws)