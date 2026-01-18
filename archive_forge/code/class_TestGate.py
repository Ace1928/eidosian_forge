from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class TestGate(cirq.Gate):

    def _qid_shape_(self):
        return (1, 2, 3)

    def _decompose_(self, qubits):
        return (cirq.X ** 0.1).on(qubits[1])