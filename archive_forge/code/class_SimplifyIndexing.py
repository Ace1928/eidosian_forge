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
class SimplifyIndexing(V.WrapperHandler):
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges):
        super().__init__(inner)
        self.name = 'SimplifyIndexing'
        self._simplify: Callable[[Expr], Expr] = lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)