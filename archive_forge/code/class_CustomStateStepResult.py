from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING
import numpy as np
from cirq import sim
from cirq.sim.simulation_state import TSimulationState
class CustomStateStepResult(sim.StepResultBase[TSimulationState], Generic[TSimulationState]):
    """The step result provided by `CustomStateSimulator.simulate_moment_steps`."""