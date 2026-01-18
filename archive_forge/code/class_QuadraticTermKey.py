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
class QuadraticTermKey:
    """An id-ordered pair of variables used as a key for quadratic terms."""
    __slots__ = ('_first_var', '_second_var')

    def __init__(self, a: 'Variable', b: 'Variable'):
        """Variables a and b will be ordered internally by their ids."""
        self._first_var: 'Variable' = a
        self._second_var: 'Variable' = b
        if self._first_var.id > self._second_var.id:
            self._first_var, self._second_var = (self._second_var, self._first_var)

    @property
    def first_var(self) -> 'Variable':
        return self._first_var

    @property
    def second_var(self) -> 'Variable':
        return self._second_var

    def __eq__(self, other: 'QuadraticTermKey') -> bool:
        return bool(self._first_var == other._first_var and self._second_var == other._second_var)

    def __hash__(self) -> int:
        return hash((self._first_var, self._second_var))

    def __str__(self):
        return f'{self._first_var!s} * {self._second_var!s}'

    def __repr__(self):
        return f'QuadraticTermKey({self._first_var!r}, {self._second_var!r})'