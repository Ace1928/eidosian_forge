from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
class OtherXX(cirq.testing.TwoQubitGate):

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        m = np.array([[0, 1], [1, 0]])
        return np.kron(m, m)

    def _decompose_(self, qubits):
        assert False