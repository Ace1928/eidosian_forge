import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_concatenate(self, scope, equiv_set, loc, args, kws):
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
    if axis < 0:
        axis = len(shapes[0]) + axis
    require(0 <= axis < len(shapes[0]))
    asserts = []
    new_shape = []
    if n == 1:
        shape = shapes[0]
        n = equiv_set.get_equiv_const(shapes[0])
        shape.pop(0)
        for i in range(len(shape)):
            if i == axis:
                m = equiv_set.get_equiv_const(shape[i])
                size = m * n if m and n else None
            else:
                size = self._sum_size(equiv_set, shapes[0])
        new_shape.append(size)
    else:
        for i in range(len(shapes[0])):
            if i == axis:
                size = self._sum_size(equiv_set, [shape[i] for shape in shapes])
            else:
                sizes = [shape[i] for shape in shapes]
                asserts.append(self._call_assert_equiv(scope, loc, equiv_set, sizes))
                size = sizes[0]
            new_shape.append(size)
    return ArrayAnalysis.AnalyzeResult(shape=tuple(new_shape), pre=sum(asserts, []))