from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
class DelegatingAncillaZ(cirq.Gate):

    def __init__(self, exponent=1, measure_ancilla: bool=False):
        self._exponent = exponent
        self._measure_ancilla = measure_ancilla

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        a = cirq.NamedQubit('a')
        yield cirq.CX(qubits[0], a)
        yield PhaseUsingCleanAncilla(self._exponent).on(a)
        yield cirq.CX(qubits[0], a)
        if self._measure_ancilla:
            yield cirq.measure(a)