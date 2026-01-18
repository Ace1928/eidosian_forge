from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@property
def CZ(cls) -> 'cirq.CliffordGate':
    if not hasattr(cls, '_CZ'):
        t = qis.CliffordTableau(num_qubits=2)
        t.xs = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        t.zs = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
        cls._CZ = CliffordGate.from_clifford_tableau(t)
    return cls._CZ