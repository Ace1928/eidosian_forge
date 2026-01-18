import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_stack(self, scope, equiv_set, loc, args, kws):
    assert len(args) > 0
    loc = args[0].loc
    seq, op = find_build_sequence(self.func_ir, args[0])
    n = len(seq)
    require(n > 0)
    axis = 0
    if 'axis' in kws:
        if isinstance(kws['axis'], int):
            axis = kws['axis']
        else:
            axis = find_const(self.func_ir, kws['axis'])
    elif len(args) > 1:
        axis = find_const(self.func_ir, args[1])
    require(isinstance(axis, int))
    require(op == 'build_tuple')
    shapes = [equiv_set._get_shape(x) for x in seq]
    asserts = self._call_assert_equiv(scope, loc, equiv_set, seq)
    shape = shapes[0]
    if axis < 0:
        axis = len(shape) + axis + 1
    require(0 <= axis <= len(shape))
    new_shape = list(shape[0:axis]) + [n] + list(shape[axis:])
    return ArrayAnalysis.AnalyzeResult(shape=tuple(new_shape), pre=asserts)