import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
@value.value_equality
class MPSState(SimulationState[_MPSHandler]):
    """A state of the MPS simulation."""

    def __init__(self, *, qubits: Sequence['cirq.Qid'], prng: np.random.RandomState, simulation_options: MPSOptions=MPSOptions(), grouping: Optional[Dict['cirq.Qid', int]]=None, initial_state: int=0, classical_data: Optional['cirq.ClassicalDataStore']=None):
        """Creates and MPSState

        Args:
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: A random number generator, used to simulate measurements.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.
            initial_state: An integer representing the initial state.
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: If the grouping does not cover the qubits.
        """
        qubit_map = {q: i for i, q in enumerate(qubits)}
        final_grouping = qubit_map if grouping is None else grouping
        if final_grouping.keys() != qubit_map.keys():
            raise ValueError('Grouping must cover exactly the qubits.')
        state = _MPSHandler.create(initial_state=initial_state, qid_shape=tuple((q.dimension for q in qubits)), simulation_options=simulation_options, grouping={qubit_map[k]: v for k, v in final_grouping.items()})
        super().__init__(state=state, prng=prng, qubits=qubits, classical_data=classical_data)

    def i_str(self, i: int) -> str:
        return self._state.i_str(i)

    def mu_str(self, i: int, j: int) -> str:
        return self._state.mu_str(i, j)

    def __str__(self) -> str:
        return str(self._state)

    def _value_equality_values_(self) -> Any:
        return (self.qubits, self._state)

    def state_vector(self) -> np.ndarray:
        """Returns the full state vector.

        Returns:
            A vector that contains the full state.
        """
        return self._state.state_vector()

    def partial_trace(self, keep_qubits: Set['cirq.Qid']) -> np.ndarray:
        """Traces out all qubits except keep_qubits.

        Args:
            keep_qubits: The set of qubits that are left after computing the
                partial trace. For example, if we have a circuit for 3 qubits
                and this parameter only has one qubit, the entire density matrix
                would be 8x8, but this function returns a 2x2 matrix.

        Returns:
            An array that contains the partial trace.
        """
        return self._state.partial_trace(set(self.get_axes(list(keep_qubits))))

    def to_numpy(self) -> np.ndarray:
        """An alias for the state vector."""
        return self._state.to_numpy()

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool=True) -> bool:
        """Delegates the action to self.apply_op"""
        return self._state.apply_op(action, self.get_axes(qubits), self.prng)

    def estimation_stats(self):
        """Returns some statistics about the memory usage and quality of the approximation."""
        return self._state.estimation_stats()

    @property
    def M(self):
        return self._state._M