import functools
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def _simplify_loops_impl(self, index_vars: List[sympy.Symbol], sizes, index_formulas):
    """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
    sizes = list(map(self.simplify, sizes))
    strides = [self.stride_vars(x, index_vars) for x in index_formulas]
    assert len(sizes) == len(strides[0]), (len(sizes), len(strides[0]))
    for i in range(len(sizes)):
        if sizes[i] == 1:
            sizes[i] = None

    def can_merge_dims(a, b):
        for k in range(len(strides)):
            if self.simplify(strides[k][a] * sizes[a]) == self.simplify(strides[k][b]):
                va = index_vars[a]
                vb = index_vars[b]
                v = sympy_symbol('_merge_tester')
                expr1 = sympy_subs(index_formulas[k], {va: v * sizes[a], vb: 0})
                expr2 = sympy_subs(index_formulas[k], {va: 0, vb: v})
                if self.simplify(expr1) == self.simplify(expr2):
                    continue
            return False
        return True
    changed = True
    while changed:
        changed = False
        for i, j in itertools.product(reversed(range(len(sizes))), reversed(range(len(sizes)))):
            if i == j or sizes[i] is None or sizes[j] is None:
                continue
            if can_merge_dims(i, j):
                changed = True
                sizes[i] = sizes[i] * sizes[j]
                sizes[j] = None

    def reindex(index):
        it = list(reversed(index))
        new_index = []
        for size in sizes:
            if size is None:
                new_index.append(sympy.Integer(0))
            else:
                new_index.append(it.pop())
        assert not it
        return new_index

    def prune(index):
        assert len(index) == len(sizes)
        return [i for i, s in zip(index, sizes) if s is not None]
    return ([x for x in sizes if x is not None], reindex, prune)