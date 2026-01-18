import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_random_laplace(self, scope, equiv_set, loc, args, kws):
    return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)