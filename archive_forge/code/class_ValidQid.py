from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class ValidQid(cirq.Qid):

    def __init__(self, name, dimension):
        self._name = name
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def dimension(self):
        return self._dimension

    def with_dimension(self, dimension):
        return ValidQid(self._name, dimension)

    def _comparison_key(self):
        return self._name