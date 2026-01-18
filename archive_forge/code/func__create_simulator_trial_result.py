from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING
import numpy as np
from cirq import sim
from cirq.sim.simulation_state import TSimulationState
def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[TSimulationState]') -> 'CustomStateTrialResult[TSimulationState]':
    return CustomStateTrialResult(params, measurements, final_simulator_state=final_simulator_state)