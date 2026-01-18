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
class QuadraticTerm(QuadraticBase):
    """The product of a scalar and two variables.

    This class is immutable.
    """
    __slots__ = ('_key', '_coefficient')

    def __init__(self, key: QuadraticTermKey, coefficient: float) -> None:
        self._key: QuadraticTermKey = key
        self._coefficient: float = coefficient

    @property
    def key(self) -> QuadraticTermKey:
        return self._key

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def _quadratic_flatten_once_and_add_to(self, scale: float, processed_elements: _QuadraticProcessedElements, target_stack: _ToProcessElements) -> None:
        processed_elements.quadratic_terms[self._key] += self._coefficient * scale

    def __mul__(self, constant: float) -> 'QuadraticTerm':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return QuadraticTerm(self._key, self._coefficient * constant)

    def __rmul__(self, constant: float) -> 'QuadraticTerm':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return QuadraticTerm(self._key, self._coefficient * constant)

    def __truediv__(self, constant: float) -> 'QuadraticTerm':
        if not isinstance(constant, (int, float)):
            return NotImplemented
        return QuadraticTerm(self._key, self._coefficient / constant)

    def __neg__(self) -> 'QuadraticTerm':
        return QuadraticTerm(self._key, -self._coefficient)

    def __str__(self):
        return f'{self._coefficient} * {self._key!s}'

    def __repr__(self):
        return f'QuadraticTerm({self._key!r}, {self._coefficient})'