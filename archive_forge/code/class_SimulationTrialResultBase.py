import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
class SimulationTrialResultBase(SimulationTrialResult[SimulationStateBase[TSimulationState]], Generic[TSimulationState], abc.ABC):
    """A base class for trial results."""

    def __init__(self, params: study.ParamResolver, measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[TSimulationState]') -> None:
        """Initializes the `SimulationTrialResultBase` class.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. Measurement results are a numpy ndarray of actual
                boolean measurement results (ordered by the qubits acted on by
                the measurement gate.)
            final_simulator_state: The final simulator state of the system after the
                trial finishes.
        """
        super().__init__(params, measurements, final_simulator_state=final_simulator_state)
        self._merged_sim_state_cache: Optional[TSimulationState] = None

    def get_state_containing_qubit(self, qubit: 'cirq.Qid') -> TSimulationState:
        """Returns the independent state space containing the qubit.

        Args:
            qubit: The qubit whose state space is required.

        Returns:
            The state space containing the qubit."""
        return self._final_simulator_state[qubit]

    def _get_substates(self) -> Sequence[TSimulationState]:
        state = self._final_simulator_state
        if isinstance(state, SimulationProductState):
            substates: Dict[TSimulationState, int] = {}
            for q in state.qubits:
                substates[self.get_state_containing_qubit(q)] = 0
            substates[state[None]] = 0
            return tuple(substates.keys())
        return [state.create_merged_state()]

    def _get_merged_sim_state(self) -> TSimulationState:
        if self._merged_sim_state_cache is None:
            self._merged_sim_state_cache = self._final_simulator_state.create_merged_state()
        return self._merged_sim_state_cache