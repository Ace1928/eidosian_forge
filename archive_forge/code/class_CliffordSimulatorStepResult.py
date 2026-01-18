from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
class CliffordSimulatorStepResult(simulator_base.StepResultBase['cirq.StabilizerChFormSimulationState']):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, sim_state: 'cirq.SimulationStateBase[clifford.StabilizerChFormSimulationState]'):
        """Results of a step of the simulator.
        Attributes:
            sim_state: The qubit:SimulationState lookup for this step.
        """
        super().__init__(sim_state)
        self._clifford_state = None

    def __str__(self) -> str:

        def bitstring(vals):
            return ''.join(('1' if v else '0' for v in vals))
        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])
        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results]) + '\n'
        final = self.state
        return f'{measurements}{final}'

    def _repr_pretty_(self, p, cycle):
        """iPython (Jupyter) pretty print."""
        p.text('cirq.CliffordSimulatorStateResult(...)' if cycle else self.__str__())

    @property
    def state(self):
        if self._clifford_state is None:
            clifford_state = CliffordState(self._qubit_mapping)
            clifford_state.ch_form = self._merged_sim_state.state.copy()
            self._clifford_state = clifford_state
        return self._clifford_state