import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
class _(ArrayComprehensionMap):

    def __new__(cls, *args, **kwargs):
        return ArrayComprehensionMap(self._lambda, *args, **kwargs)