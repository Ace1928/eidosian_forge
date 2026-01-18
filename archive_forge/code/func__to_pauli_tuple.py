from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def _to_pauli_tuple(matrix: np.ndarray) -> Optional[Tuple[Pauli, bool]]:
    """Converts matrix to Pauli gate.

    If matrix is not ±Pauli matrix, returns None.
    """
    for pauli in Pauli._XYZ:
        p = protocols.unitary(pauli)
        if np.allclose(matrix, p):
            return (pauli, False)
        if np.allclose(matrix, -p):
            return (pauli, True)
    return None