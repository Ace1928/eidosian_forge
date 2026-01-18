from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
class CompositeExample2(cirq.testing.TwoQubitGate):

    def _decompose_(self, qubits):
        yield cirq.CZ(qubits[0], qubits[1])
        yield CompositeExample()(qubits[1])