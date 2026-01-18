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
class LinearTerm(LinearBase):
    """The product of a scalar and a variable.

    This class is immutable.
    """
    __slots__ = ('_variable', '_coefficient')

    def __init__(self, variable: Variable, coefficient: float) -> None:
        self._variable: Variable = variable
        self._coefficient: float = coefficient

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def _flatten_once_and_add_to(self, scale: float, processed_elements: _ProcessedElements, target_stack: _ToProcessElements) -> None:
        processed_elements.terms[self._variable] += self._coefficient * scale

    @typing.overload
    def __mul__(self, other: float) -> 'LinearTerm':
        ...

    @typing.overload
    def __mul__(self, other: Union['Variable', 'LinearTerm']) -> 'QuadraticTerm':
        ...

    @typing.overload
    def __mul__(self, other: 'LinearBase') -> 'LinearLinearProduct':
        ...

    def __mul__(self, other):
        if not isinstance(other, (int, float, LinearBase)):
            return NotImplemented
        if isinstance(other, Variable):
            return QuadraticTerm(QuadraticTermKey(self._variable, other), self._coefficient)
        if isinstance(other, LinearTerm):
            return QuadraticTerm(QuadraticTermKey(self.variable, other.variable), self._coefficient * other.coefficient)
        if isinstance(other, LinearBase):
            return LinearLinearProduct(self, other)
        return LinearTerm(self._variable, self._coefficient * other)

    def __rmul__(self, constant: float) -> 'LinearTerm':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return LinearTerm(self._variable, self._coefficient * constant)

    def __truediv__(self, constant: float) -> 'LinearTerm':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return LinearTerm(self._variable, self._coefficient / constant)

    def __neg__(self) -> 'LinearTerm':
        return LinearTerm(self._variable, -self._coefficient)

    def __str__(self):
        return f'{self._coefficient} * {self._variable}'

    def __repr__(self):
        return f'LinearTerm({self._variable!r}, {self._coefficient!r})'