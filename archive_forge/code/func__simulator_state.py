import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _simulator_state(self) -> TSimulatorState:
    """Returns the simulator state of the simulator after this step.

        This method starts with an underscore to indicate that it is private.
        To access public state, see public methods on StepResult.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """
    return self._sim_state