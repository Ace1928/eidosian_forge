import pytest
import numpy as np
import sympy
import cirq
class ParameterizedGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        yield (cirq.X(qubits[0]) ** sympy.Symbol('x'))
        yield (cirq.Y(qubits[1]) ** sympy.Symbol('y'))