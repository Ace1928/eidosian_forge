import pytest
import numpy as np
import sympy
import cirq
class GateDecomposeNotImplemented(cirq.testing.SingleQubitGate):

    def _decompose_(self, qubits):
        return NotImplemented