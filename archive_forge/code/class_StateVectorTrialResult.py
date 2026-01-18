import abc
from typing import Any, Dict, Iterator, Sequence, Type, TYPE_CHECKING, Generic, TypeVar
import numpy as np
from cirq import _compat, ops, value, qis
from cirq.sim import simulator, state_vector, simulator_base
from cirq.protocols import qid_shape
@value.value_equality(unhashable=True)
class StateVectorTrialResult(state_vector.StateVectorMixin, simulator_base.SimulationTrialResultBase['cirq.StateVectorSimulationState']):
    """A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

    Attributes:
        final_state_vector: The final state vector for the system.
    """

    def __init__(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]') -> None:
        super().__init__(params=params, measurements=measurements, final_simulator_state=final_simulator_state, qubit_map=final_simulator_state.qubit_map)

    @_compat.cached_property
    def final_state_vector(self) -> np.ndarray:
        return self._get_merged_sim_state().target_tensor.reshape(-1)

    def state_vector(self, copy: bool=False) -> np.ndarray:
        """Return the state vector at the end of the computation.

        The state is returned in the computational basis with these basis
        states defined by the qubit_map. In particular the value in the
        qubit_map is the index of the qubit, and these are translated into
        binary vectors where the last qubit is the 1s bit of the index, the
        second-to-last is the 2s bit of the index, and so forth (i.e. big
        endian ordering).

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned vector will have indices mapped to qubit basis
             states like the following table

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |

        Args:
            copy: If True, the returned state vector will be a copy of that
            stored by the object. This is potentially expensive for large
            state vectors, but prevents mutation of the object state, e.g. for
            operating on intermediate states of a circuit.
            Defaults to False.
        """
        return self.final_state_vector.copy() if copy else self.final_state_vector

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self.qubit_map, self.final_state_vector.tolist())

    def __str__(self) -> str:
        samples = super().__str__()
        ret = f'measurements: {samples}'
        for substate in self._get_substates():
            final = substate.target_tensor
            shape = final.shape
            size = np.prod(shape, dtype=np.int64)
            final = final.reshape(size)
            if len([1 for e in final if abs(e) > 0.001]) < 16:
                state_vector = qis.dirac_notation(final, 3, qid_shape(substate.qubits))
            else:
                state_vector = str(final)
            label = f'qubits: {substate.qubits}' if substate.qubits else 'phase:'
            ret += f'\n\n{label}\noutput vector: {state_vector}'
        return ret

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        if cycle:
            p.text('StateVectorTrialResult(...)')
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        return f'cirq.StateVectorTrialResult(params={self.params!r}, measurements={_compat.proper_repr(self.measurements)}, final_simulator_state={self._final_simulator_state!r})'