from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
class CliffordTrialResult(simulator_base.SimulationTrialResultBase['clifford.StabilizerChFormSimulationState']):

    def __init__(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.StabilizerChFormSimulationState]') -> None:
        super().__init__(params=params, measurements=measurements, final_simulator_state=final_simulator_state)

    @property
    def final_state(self) -> 'cirq.CliffordState':
        state = self._get_merged_sim_state()
        clifford_state = CliffordState(state.qubit_map)
        clifford_state.ch_form = state.state.copy()
        return clifford_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._get_merged_sim_state().state
        return f'measurements: {samples}\noutput state: {final}'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text('cirq.CliffordTrialResult(...)' if cycle else self.__str__())