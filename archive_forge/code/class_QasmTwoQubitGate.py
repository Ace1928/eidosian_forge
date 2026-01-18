from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
@value.value_equality
class QasmTwoQubitGate(ops.Gate):

    def __init__(self, kak: 'cirq.KakDecomposition') -> None:
        """A two qubit gate represented in QASM by the KAK decomposition.

        All angles are in half turns.  Assumes a canonicalized KAK
        decomposition.

        Args:
            kak: KAK decomposition of the two-qubit gate.
        """
        self.kak = kak

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self):
        return self.kak

    @staticmethod
    def from_matrix(mat: np.ndarray, atol=1e-08) -> 'QasmTwoQubitGate':
        """Creates a QasmTwoQubitGate from the given matrix.

        Args:
            mat: The unitary matrix of the two qubit gate.
            atol: Absolute error tolerance when decomposing.

        Returns:
            A QasmTwoQubitGate implementing the matrix.
        """
        kak = linalg.kak_decomposition(mat, atol=atol)
        return QasmTwoQubitGate(kak)

    def _unitary_(self):
        return protocols.unitary(self.kak)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        q0, q1 = qubits
        x, y, z = self.kak.interaction_coefficients
        a = x * -2 / np.pi + 0.5
        b = y * -2 / np.pi + 0.5
        c = z * -2 / np.pi + 0.5
        b0, b1 = self.kak.single_qubit_operations_before
        yield QasmUGate.from_matrix(b0).on(q0)
        yield QasmUGate.from_matrix(b1).on(q1)
        yield (ops.X(q0) ** 0.5)
        yield ops.CNOT(q0, q1)
        yield (ops.X(q0) ** a)
        yield (ops.Y(q1) ** b)
        yield ops.CNOT(q1, q0)
        yield (ops.X(q1) ** (-0.5))
        yield (ops.Z(q1) ** c)
        yield ops.CNOT(q0, q1)
        a0, a1 = self.kak.single_qubit_operations_after
        yield QasmUGate.from_matrix(a0).on(q0)
        yield QasmUGate.from_matrix(a1).on(q1)

    def __repr__(self) -> str:
        return f'cirq.circuits.qasm_output.QasmTwoQubitGate({self.kak!r})'