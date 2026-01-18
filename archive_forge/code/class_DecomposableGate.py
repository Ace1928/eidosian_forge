from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class DecomposableGate(cirq.Gate):

    def __init__(self, unitary_value):
        self.unitary_value = unitary_value

    def num_qubits(self):
        return 1

    def _decompose_(self, qubits):
        yield FullyImplemented(self.unitary_value).on(qubits[0])