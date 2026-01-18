import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def _base_iterator(self, circuit: 'cirq.AbstractCircuit', qubits: Tuple['cirq.Qid', ...], initial_state: Any) -> Iterator[TStepResultBase]:
    sim_state = self._create_simulation_state(initial_state, qubits)
    return self._core_iterator(circuit, sim_state)