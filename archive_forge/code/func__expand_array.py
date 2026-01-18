import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
def _expand_array(self):
    res = []
    for values in itertools.product(*[range(inf, sup + 1) for var, inf, sup in self._limits]):
        res.append(self._get_element(values))
    return ImmutableDenseNDimArray(res, self.shape)