import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
def _as_pauli_mask(val: Union[Iterable['cirq.PAULI_GATE_LIKE'], np.ndarray]) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return np.asarray(val, dtype=np.uint8)
    return np.array([_pauli_index(v) for v in val], dtype=np.uint8)