import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class CountingSimulationState(cirq.SimulationState[CountingState]):

    def __init__(self, state, qubits, classical_data):
        state_obj = CountingState(state)
        super().__init__(state=state_obj, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool=True) -> bool:
        self._state.gate_count += 1
        return True

    @property
    def data(self):
        return self._state.data

    @property
    def gate_count(self):
        return self._state.gate_count

    @property
    def measurement_count(self):
        return self._state.measurement_count

    @property
    def copy_count(self):
        return self._state.copy_count