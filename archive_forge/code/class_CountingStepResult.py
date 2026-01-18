import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class CountingStepResult(cirq.StepResultBase[CountingSimulationState]):

    def sample(self, qubits: List[cirq.Qid], repetitions: int=1, seed: cirq.RANDOM_STATE_OR_SEED_LIKE=None) -> np.ndarray:
        measurements: List[List[int]] = []
        for _ in range(repetitions):
            measurements.append(self._merged_sim_state._perform_measurement(qubits))
        return np.array(measurements, dtype=int)

    def _simulator_state(self) -> CountingSimulationState:
        return self._merged_sim_state