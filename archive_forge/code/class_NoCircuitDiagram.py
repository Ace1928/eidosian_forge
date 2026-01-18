from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NoCircuitDiagram(cirq.Gate):

    def num_qubits(self) -> int:
        return 1

    def __repr__(self):
        return 'guess-i-will-repr'