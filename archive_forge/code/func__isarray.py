import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _isarray(self, varname):
    typ = self.typemap[varname]
    return isinstance(typ, types.npytypes.Array) and typ.ndim > 0