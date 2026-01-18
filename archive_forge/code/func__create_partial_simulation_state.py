from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def _create_partial_simulation_state(self, initial_state: Union[int, cirq.StateVectorSimulationState], qubits: Sequence[cirq.Qid], classical_data: cirq.ClassicalDataStore) -> cirq.StateVectorSimulationState:
    raise NotImplementedError()