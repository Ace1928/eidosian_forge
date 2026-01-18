from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NoActOn(cirq.Gate):

    def _num_qubits_(self) -> int:
        return 1

    def _act_on_(self, sim_state, qubits):
        return NotImplemented