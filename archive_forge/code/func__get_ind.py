import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _get_ind(self, x):
    """Return the internal index (greater or equal to 0) of the given
        object, or -1 if not found.
        """
    return self.obj_to_ind.get(x, -1)