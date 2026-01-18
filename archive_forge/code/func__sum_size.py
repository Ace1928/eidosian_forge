import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _sum_size(self, equiv_set, sizes):
    """Return the sum of the given list of sizes if they are all equivalent
        to some constant, or None otherwise.
        """
    s = 0
    for size in sizes:
        n = equiv_set.get_equiv_const(size)
        if n is None:
            return None
        else:
            s += n
    return s