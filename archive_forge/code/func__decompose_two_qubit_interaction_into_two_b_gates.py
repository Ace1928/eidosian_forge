from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _decompose_two_qubit_interaction_into_two_b_gates(interaction: Union['cirq.SupportsUnitary', np.ndarray, 'cirq.KakDecomposition'], *, qubits: Sequence['cirq.Qid']) -> List['cirq.Operation']:
    kak = linalg.kak_decomposition(interaction)
    result = _decompose_interaction_into_two_b_gates_ignoring_single_qubit_ops(qubits, kak.interaction_coefficients)
    return list(_fix_single_qubit_gates_around_kak_interaction(desired=kak, qubits=qubits, operations=result))