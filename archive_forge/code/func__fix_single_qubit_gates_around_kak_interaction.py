from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _fix_single_qubit_gates_around_kak_interaction(*, desired: 'cirq.KakDecomposition', operations: List['cirq.Operation'], qubits: Sequence['cirq.Qid']) -> Iterator['cirq.Operation']:
    """Adds single qubit operations to complete a desired interaction.

    Args:
        desired: The kak decomposition of the desired operation.
        qubits: The pair of qubits that is being operated on.
        operations: A list of operations that composes into the desired kak
            interaction coefficients, but may not have the desired before/after
            single qubit operations or the desired global phase.

    Returns:
        A list of operations whose kak decomposition approximately equals the
        desired kak decomposition.
    """
    actual = linalg.kak_decomposition(circuits.Circuit(operations).unitary(qubit_order=qubits))

    def dag(a: np.ndarray) -> np.ndarray:
        return np.transpose(np.conjugate(a))
    for k in range(2):
        g = ops.MatrixGate(dag(actual.single_qubit_operations_before[k]) @ desired.single_qubit_operations_before[k])
        yield g(qubits[k])
    yield from operations
    for k in range(2):
        g = ops.MatrixGate(desired.single_qubit_operations_after[k] @ dag(actual.single_qubit_operations_after[k]))
        yield g(qubits[k])
    yield ops.global_phase_operation(desired.global_phase / actual.global_phase)