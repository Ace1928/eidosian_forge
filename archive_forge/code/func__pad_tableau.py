from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def _pad_tableau(clifford_tableau: qis.CliffordTableau, num_qubits_after_padding: int, axes: List[int]) -> qis.CliffordTableau:
    """Roughly, this function copies self.tableau into the "identity" matrix."""
    if len(set(axes)) != clifford_tableau.n:
        raise ValueError('Input axes of padding should match with the number of qubits in the input tableau.')
    if clifford_tableau.n > num_qubits_after_padding:
        raise ValueError('The number of qubits in the input tableau should not be larger than num_qubits_after_padding.')
    padded_tableau = qis.CliffordTableau(num_qubits_after_padding)
    v_index = np.concatenate((np.asarray(axes), num_qubits_after_padding + np.asarray(axes)))
    padded_tableau.xs[np.ix_(v_index, axes)] = clifford_tableau.xs
    padded_tableau.zs[np.ix_(v_index, axes)] = clifford_tableau.zs
    padded_tableau.rs[v_index] = clifford_tableau.rs
    return padded_tableau