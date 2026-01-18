import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _imul_helper_checkpoint(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
    """Like `_imul_helper` but guarantees no-op on error."""
    if not isinstance(other, (numbers.Number, PauliString, MutablePauliString)):
        other = MutablePauliString()._imul_helper(other, sign)
        if other is NotImplemented:
            return NotImplemented
    return self._imul_helper(other, sign)