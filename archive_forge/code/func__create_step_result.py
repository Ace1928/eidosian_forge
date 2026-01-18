from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def _create_step_result(self, sim_state: cirq.SimulationStateBase) -> cirq.SparseSimulatorStep:
    raise NotImplementedError()