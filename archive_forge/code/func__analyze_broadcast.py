import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_broadcast(self, scope, equiv_set, loc, args, fn):
    """Infer shape equivalence of arguments based on Numpy broadcast rules
        and return shape of output
        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        """
    tups = list(filter(lambda a: self._istuple(a.name), args))
    if len(tups) == 2 and fn.__name__ == 'add':
        tup0typ = self.typemap[tups[0].name]
        tup1typ = self.typemap[tups[1].name]
        if tup0typ.count == 0:
            return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[1]))
        if tup1typ.count == 0:
            return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[0]))
        try:
            shapes = [equiv_set.get_shape(x) for x in tups]
            if None in shapes:
                return None
            concat_shapes = sum(shapes, ())
            return ArrayAnalysis.AnalyzeResult(shape=concat_shapes)
        except GuardException:
            return None
    arrs = list(filter(lambda a: self._isarray(a.name), args))
    require(len(arrs) > 0)
    names = [x.name for x in arrs]
    dims = [self.typemap[x.name].ndim for x in arrs]
    max_dim = max(dims)
    require(max_dim > 0)
    try:
        shapes = [equiv_set.get_shape(x) for x in arrs]
    except GuardException:
        return ArrayAnalysis.AnalyzeResult(shape=arrs[0], pre=self._call_assert_equiv(scope, loc, equiv_set, arrs))
    pre = []
    if None in shapes:
        new_shapes = []
        for i, s in enumerate(shapes):
            if s is None:
                var = arrs[i]
                typ = self.typemap[var.name]
                shape = self._gen_shape_call(equiv_set, var, typ.ndim, None, pre)
                new_shapes.append(shape)
            else:
                new_shapes.append(s)
        shapes = new_shapes
    result = self._broadcast_assert_shapes(scope, equiv_set, loc, shapes, names)
    if pre:
        if 'pre' in result.kwargs:
            prev_pre = result.kwargs['pre']
        else:
            prev_pre = []
        result.kwargs['pre'] = pre + prev_pre
    return result