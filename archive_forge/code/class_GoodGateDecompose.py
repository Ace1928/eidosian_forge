import pytest
import numpy as np
import sympy
import cirq
class GoodGateDecompose(cirq.testing.SingleQubitGate):

    def _decompose_(self, qubits):
        return cirq.X(qubits[0])

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])