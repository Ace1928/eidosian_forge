from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def _approx_eq_(self, other, atol):
    if not isinstance(self.val, type(other)):
        return NotImplemented
    return cirq.approx_eq(self.val, other, atol=atol)