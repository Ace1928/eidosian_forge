import cirq
import pytest
import numpy as np
class FailsOnDecompostion(cirq.Gate):

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def _has_unitary_(self) -> bool:
        return True

    def _decompose_with_context_(self, qubits, *, context):
        q, = context.qubit_manager.qalloc(1)
        yield cirq.X(q)
        yield cirq.measure(qubits[0])