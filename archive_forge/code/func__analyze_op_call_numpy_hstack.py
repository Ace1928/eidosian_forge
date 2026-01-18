import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_hstack(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1
    seq, op = find_build_sequence(self.func_ir, args[0])
    n = len(seq)
    require(n > 0)
    typ = self.typemap[seq[0].name]
    require(isinstance(typ, types.ArrayCompatible))
    if typ.ndim < 2:
        kws['axis'] = 0
    else:
        kws['axis'] = 1
    return self._analyze_op_call_numpy_concatenate(scope, equiv_set, loc, args, kws)