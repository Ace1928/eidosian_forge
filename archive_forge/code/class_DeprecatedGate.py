from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class DeprecatedGate(cirq.Gate):

    def num_qubits(self):
        return 3