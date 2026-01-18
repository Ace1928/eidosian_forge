from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def _to_clifford_tableau(rotation_map: Optional[Dict[Pauli, Tuple[Pauli, bool]]]=None, *, x_to: Optional[Tuple[Pauli, bool]]=None, z_to: Optional[Tuple[Pauli, bool]]=None) -> qis.CliffordTableau:
    """Transfer the rotation map to clifford tableau representation"""
    if x_to is None and z_to is None and (rotation_map is None):
        raise ValueError('The function either takes rotation_map or a combination  of x_to and z_to but none were given.')
    elif rotation_map is not None:
        x_to = rotation_map[pauli_gates.X]
        z_to = rotation_map[pauli_gates.Z]
    else:
        assert x_to is not None and z_to is not None, 'Both x_to and z_to have to be provided.'
    clifford_tableau = qis.CliffordTableau(num_qubits=1)
    clifford_tableau.xs[0, 0] = x_to[0] in (pauli_gates.X, pauli_gates.Y)
    clifford_tableau.zs[0, 0] = x_to[0] in (pauli_gates.Y, pauli_gates.Z)
    clifford_tableau.xs[1, 0] = z_to[0] in (pauli_gates.X, pauli_gates.Y)
    clifford_tableau.zs[1, 0] = z_to[0] in (pauli_gates.Y, pauli_gates.Z)
    clifford_tableau.rs = np.array([x_to[1], z_to[1]])
    return clifford_tableau