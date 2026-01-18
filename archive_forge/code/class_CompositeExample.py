from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
class CompositeExample(cirq.testing.SingleQubitGate):

    def _decompose_(self, qubits):
        yield cirq.X(qubits[0])
        yield (cirq.Y(qubits[0]) ** 0.5)