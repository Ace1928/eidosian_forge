import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
class ArrayComprehensionMap(ArrayComprehension):
    """
    A subclass of ArrayComprehension dedicated to map external function lambda.

    Notes
    =====

    Only the lambda function is considered.
    At most one argument in lambda function is accepted in order to avoid ambiguity
    in value assignment.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehensionMap
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehensionMap(lambda: 1, (i, 1, 4))
    >>> a.doit()
    [1, 1, 1, 1]
    >>> b = ArrayComprehensionMap(lambda a: a+1, (j, 1, 4))
    >>> b.doit()
    [2, 3, 4, 5]

    """

    def __new__(cls, function, *symbols, **assumptions):
        if any((len(l) != 3 or None for l in symbols)):
            raise ValueError('ArrayComprehension requires values lower and upper bound for the expression')
        if not isLambda(function):
            raise ValueError('Data type not supported')
        arglist = cls._check_limits_validity(function, symbols)
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        obj._lambda = function
        return obj

    @property
    def func(self):

        class _(ArrayComprehensionMap):

            def __new__(cls, *args, **kwargs):
                return ArrayComprehensionMap(self._lambda, *args, **kwargs)
        return _

    def _get_element(self, values):
        temp = self._lambda
        if self._lambda.__code__.co_argcount == 0:
            temp = temp()
        elif self._lambda.__code__.co_argcount == 1:
            temp = temp(functools.reduce(lambda a, b: a * b, values))
        return temp