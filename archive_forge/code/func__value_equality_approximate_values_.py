import abc
import fractions
import math
import numbers
from typing import (
import numpy as np
import sympy
from cirq import value, protocols
from cirq.linalg import tolerance
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
def _value_equality_approximate_values_(self):
    period = self._period()
    if not period or protocols.is_parameterized(self._exponent):
        exponent = self._exponent
    else:
        exponent = value.PeriodicValue(self._exponent, period)
    return (exponent, self._global_shift)