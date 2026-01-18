import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class SplittableCountingSimulator(CountingSimulator):

    def __init__(self, noise=None, split_untangled_states=True):
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_partial_simulation_state(self, initial_state: Any, qubits: Sequence['cirq.Qid'], classical_data: cirq.ClassicalDataStore) -> CountingSimulationState:
        return SplittableCountingSimulationState(qubits=qubits, state=initial_state, classical_data=classical_data)