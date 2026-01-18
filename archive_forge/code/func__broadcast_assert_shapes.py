import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _broadcast_assert_shapes(self, scope, equiv_set, loc, shapes, names):
    """Produce assert_equiv for sizes in each dimension, taking into
        account of dimension coercion and constant size of 1.
        """
    asserts = []
    new_shape = []
    max_dim = max([len(shape) for shape in shapes])
    const_size_one = None
    for i in range(max_dim):
        sizes = []
        size_names = []
        for name, shape in zip(names, shapes):
            if i < len(shape):
                size = shape[len(shape) - 1 - i]
                const_size = equiv_set.get_equiv_const(size)
                if const_size == 1:
                    const_size_one = size
                else:
                    sizes.append(size)
                    size_names.append(name)
        if sizes == []:
            assert const_size_one is not None
            sizes.append(const_size_one)
            size_names.append('1')
        asserts.append(self._call_assert_equiv(scope, loc, equiv_set, sizes, names=size_names))
        new_shape.append(sizes[0])
    return ArrayAnalysis.AnalyzeResult(shape=tuple(reversed(new_shape)), pre=sum(asserts, []))