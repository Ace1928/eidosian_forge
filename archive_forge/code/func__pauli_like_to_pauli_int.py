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
def _pauli_like_to_pauli_int(key: Any, pauli_gate_like: PAULI_GATE_LIKE):
    pauli_int = PAULI_GATE_LIKE_TO_INDEX_MAP.get(pauli_gate_like, None)
    if pauli_int is None:
        raise TypeError(f"Expected {key!r}: {pauli_gate_like!r} to have a cirq.PAULI_GATE_LIKE value. But the value isn't in {set(PAULI_GATE_LIKE_TO_INDEX_MAP.keys())!r}")
    return pauli_int