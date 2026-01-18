import numpy as np
import pytest
import sympy
import cirq
class CustomCnotGate(cirq.Gate):

    def num_qubits(self) -> int:
        return 2

    def _unitary_(self):
        return cirq.unitary(cirq.CNOT)