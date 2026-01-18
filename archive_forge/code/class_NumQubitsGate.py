import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class NumQubitsGate(cirq.Gate):

    def _num_qubits_(self):
        return 4