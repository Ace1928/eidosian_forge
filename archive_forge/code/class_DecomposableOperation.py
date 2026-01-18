from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class DecomposableOperation(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits, unitary_value):
        self.qubits = qubits
        self.unitary_value = unitary_value

    def _decompose_(self):
        for q in self.qubits:
            yield FullyImplemented(self.unitary_value)(q)