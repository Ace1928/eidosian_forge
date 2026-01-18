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
class QuadraticSum(QuadraticBase):
    """A deferred sum of QuadraticTypes objects.

    QuadraticSum objects are automatically created when a quadratic object is
    added to quadratic or linear objects and, as has performance optimizations
    similar to LinearSum.

    This class is immutable.
    """
    __slots__ = ('__weakref__', '_elements')

    def __init__(self, iterable: Iterable[QuadraticTypes]) -> None:
        """Creates a QuadraticSum object. A copy of iterable is saved as a tuple."""
        self._elements = tuple(iterable)
        for item in self._elements:
            if not isinstance(item, (LinearBase, QuadraticBase, int, float)):
                raise TypeError(f'unsupported type in iterable argument for QuadraticSum: {type(item).__name__!r}')

    @property
    def elements(self) -> Tuple[QuadraticTypes, ...]:
        return self._elements

    def _quadratic_flatten_once_and_add_to(self, scale: float, processed_elements: _QuadraticProcessedElements, target_stack: _QuadraticToProcessElements) -> None:
        for term in self._elements:
            if isinstance(term, (int, float)):
                processed_elements.offset += scale * float(term)
            else:
                target_stack.append(term, scale)

    def __str__(self):
        return str(as_flat_quadratic_expression(self))

    def __repr__(self):
        result = 'QuadraticSum(('
        result += ', '.join((repr(element) for element in self._elements))
        result += '))'
        return result