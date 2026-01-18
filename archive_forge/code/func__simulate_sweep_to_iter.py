import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _simulate_sweep_to_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
    if type(self).simulate_sweep == SimulatesFinalState.simulate_sweep:
        raise RecursionError('Must define either simulate_sweep or simulate_sweep_iter.')
    yield from self.simulate_sweep(program, params, qubit_order, initial_state)