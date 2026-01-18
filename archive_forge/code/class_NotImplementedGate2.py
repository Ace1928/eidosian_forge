from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NotImplementedGate2(cirq.Gate):

    def _num_qubits_(self):
        return NotImplemented