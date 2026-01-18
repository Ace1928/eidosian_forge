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
def _as_pauli_string(self) -> PauliString:
    return PauliString(qubit_pauli_map={self.qubit: self.pauli})