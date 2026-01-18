import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
class ExampleCompositeOperation(cirq.Operation):
    qubits = (q0,)
    with_qubits = NotImplemented

    def _decompose_(self):
        return cirq.X(self.qubits[0])

    def __repr__(self):
        return 'ExampleCompositeOperation()'