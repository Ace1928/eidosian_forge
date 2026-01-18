import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class UpperBoundedLinearExpression:
    """An inequality of the form expression <= upper_bound.

    Where:
     * expression is a linear expression, and
     * upper_bound is a float
    """
    __slots__ = ('_expression', '_upper_bound')

    def __init__(self, expression: 'LinearBase', upper_bound: float) -> None:
        """Operator overloading can be used instead: e.g. `x + y <= 2.0`."""
        self._expression: 'LinearBase' = expression
        self._upper_bound: float = upper_bound

    @property
    def expression(self) -> 'LinearBase':
        return self._expression

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    def __ge__(self, lhs: float) -> 'BoundedLinearExpression':
        if isinstance(lhs, (int, float)):
            return BoundedLinearExpression(lhs, self.expression, self.upper_bound)
        _raise_binary_operator_type_error('>=', type(self), type(lhs))

    def __bool__(self) -> bool:
        raise TypeError('__bool__ is unsupported for UpperBoundedLinearExpression' + '\n' + _CHAINED_COMPARISON_MESSAGE)

    def __str__(self):
        return f'{self._expression!s} <= {self._upper_bound}'

    def __repr__(self):
        return f'{self._expression!r} <= {self._upper_bound}'