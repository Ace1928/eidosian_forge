import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
class StepResultBase(Generic[TSimulationState], StepResult[SimulationStateBase[TSimulationState]], abc.ABC):
    """A base class for step results."""

    def __init__(self, sim_state: SimulationStateBase[TSimulationState]):
        """Initializes the step result.

        Args:
            sim_state: The `SimulationStateBase` for this step.
        """
        super().__init__(sim_state)
        self._merged_sim_state_cache: Optional[TSimulationState] = None
        qubits = sim_state.qubits
        self._qubits = qubits
        self._qubit_mapping = {q: i for i, q in enumerate(qubits)}
        self._qubit_shape = tuple((q.dimension for q in qubits))
        self._classical_data = sim_state.classical_data

    def _qid_shape_(self):
        return self._qubit_shape

    @property
    def _merged_sim_state(self) -> TSimulationState:
        if self._merged_sim_state_cache is None:
            self._merged_sim_state_cache = self._sim_state.create_merged_state()
        return self._merged_sim_state_cache

    def sample(self, qubits: List['cirq.Qid'], repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
        return self._sim_state.sample(qubits, repetitions, seed)