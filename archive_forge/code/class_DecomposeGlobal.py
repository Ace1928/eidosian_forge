import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class DecomposeGlobal(cirq.Gate):

    def num_qubits(self):
        return 1

    def _decompose_(self, qubits):
        yield cirq.global_phase_operation(1j)