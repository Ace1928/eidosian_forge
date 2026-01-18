from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class DecomposableOrder(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits):
        self.qubits = qubits

    def _decompose_(self):
        yield FullyImplemented(True)(self.qubits[2])
        yield FullyImplemented(True)(self.qubits[0])