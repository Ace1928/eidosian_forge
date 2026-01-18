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
class QuadraticProduct(QuadraticBase):
    """A deferred multiplication computation for quadratic expressions.

    This class is immutable.
    """
    __slots__ = ('_scalar', '_quadratic')

    def __init__(self, scalar: float, quadratic: QuadraticBase) -> None:
        if not isinstance(scalar, (float, int)):
            raise TypeError(f'unsupported type for scalar argument in QuadraticProduct: {type(scalar).__name__!r}')
        if not isinstance(quadratic, QuadraticBase):
            raise TypeError(f'unsupported type for linear argument in QuadraticProduct: {type(quadratic).__name__!r}')
        self._scalar: float = float(scalar)
        self._quadratic: QuadraticBase = quadratic

    @property
    def scalar(self) -> float:
        return self._scalar

    @property
    def quadratic(self) -> QuadraticBase:
        return self._quadratic

    def _quadratic_flatten_once_and_add_to(self, scale: float, processed_elements: _QuadraticProcessedElements, target_stack: _QuadraticToProcessElements) -> None:
        target_stack.append(self._quadratic, self._scalar * scale)

    def __str__(self):
        return str(as_flat_quadratic_expression(self))

    def __repr__(self):
        return f'QuadraticProduct({self._scalar}, {self._quadratic!r})'