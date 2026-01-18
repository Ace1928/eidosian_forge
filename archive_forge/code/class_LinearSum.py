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
class LinearSum(LinearBase):
    """A deferred sum of LinearBase objects.

    LinearSum objects are automatically created when two linear objects are added
    and, as noted in the documentation for Linear, can reduce the inefficiencies.
    In particular, they are created when calling sum(iterable) when iterable is
    an Iterable[LinearTypes]. However, using LinearSum(iterable) instead
    can result in additional performance improvements:

      * sum(iterable): creates a nested set of LinearSum objects (e.g.
        `sum([a, b, c])` is `LinearSum(0, LinearSum(a, LinearSum(b, c)))`).
      * LinearSum(iterable): creates a single LinearSum that saves a tuple with
        all the LinearTypes objects in iterable (e.g.
        `LinearSum([a, b, c])` does not create additional objects).

    This class is immutable.
    """
    __slots__ = ('__weakref__', '_elements')

    def __init__(self, iterable: Iterable[LinearTypes]) -> None:
        """Creates a LinearSum object. A copy of iterable is saved as a tuple."""
        self._elements = tuple(iterable)
        for item in self._elements:
            if not isinstance(item, (LinearBase, int, float)):
                raise TypeError(f'unsupported type in iterable argument for LinearSum: {type(item).__name__!r}')

    @property
    def elements(self) -> Tuple[LinearTypes, ...]:
        return self._elements

    def _flatten_once_and_add_to(self, scale: float, processed_elements: _ProcessedElements, target_stack: _ToProcessElements) -> None:
        for term in self._elements:
            if isinstance(term, (int, float)):
                processed_elements.offset += scale * float(term)
            else:
                target_stack.append(term, scale)

    def __str__(self):
        return str(as_flat_linear_expression(self))

    def __repr__(self):
        result = 'LinearSum(('
        result += ', '.join((repr(linear) for linear in self._elements))
        result += '))'
        return result