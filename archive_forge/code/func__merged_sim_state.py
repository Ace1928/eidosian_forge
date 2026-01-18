import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
@property
def _merged_sim_state(self) -> TSimulationState:
    if self._merged_sim_state_cache is None:
        self._merged_sim_state_cache = self._sim_state.create_merged_state()
    return self._merged_sim_state_cache