import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _simulate_expectation_values_sweep_to_iter(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> Iterator[List[float]]:
    if type(self).simulate_expectation_values_sweep == SimulatesExpectationValues.simulate_expectation_values_sweep:
        raise RecursionError('Must define either simulate_expectation_values_sweep or simulate_expectation_values_sweep_iter.')
    yield from self.simulate_expectation_values_sweep(program, observables, params, qubit_order, initial_state, permit_terminal_measurements)