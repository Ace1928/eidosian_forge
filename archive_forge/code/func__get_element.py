import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
def _get_element(self, values):
    temp = self._lambda
    if self._lambda.__code__.co_argcount == 0:
        temp = temp()
    elif self._lambda.__code__.co_argcount == 1:
        temp = temp(functools.reduce(lambda a, b: a * b, values))
    return temp