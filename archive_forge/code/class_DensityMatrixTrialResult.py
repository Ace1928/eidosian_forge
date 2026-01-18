from typing import Any, Dict, List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import ops, protocols, study, value
from cirq._compat import proper_repr
from cirq.sim import simulator, density_matrix_simulation_state, simulator_base
@value.value_equality(unhashable=True)
class DensityMatrixTrialResult(simulator_base.SimulationTrialResultBase[density_matrix_simulation_state.DensityMatrixSimulationState]):
    """A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

    The density matrix that is stored in this result is returned in the
    computational basis with these basis states defined by the qubit_map.
    In particular, the value in the qubit_map is the index of the qubit,
    and these are translated into binary vectors where the last qubit is
    the 1s bit of the index, the second-to-last is the 2s bit of the index,
    and so forth (i.e. big endian ordering). The density matrix is a
    `2 ** num_qubits` square matrix, with rows and columns ordered by
    the computational basis as just described.

    Example:
         qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
         Then the returned density matrix will have (row and column) indices
         mapped to qubit basis states like the following table

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

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_simulator_state: The final simulator state of the system after the
            trial finishes.
    """

    def __init__(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.DensityMatrixSimulationState]') -> None:
        super().__init__(params=params, measurements=measurements, final_simulator_state=final_simulator_state)
        self._final_density_matrix: Optional[np.ndarray] = None

    @property
    def final_density_matrix(self) -> np.ndarray:
        if self._final_density_matrix is None:
            size = np.prod(protocols.qid_shape(self), dtype=np.int64)
            tensor = self._get_merged_sim_state().target_tensor
            self._final_density_matrix = np.reshape(tensor.copy(), (size, size))
        return self._final_density_matrix

    def _value_equality_values_(self) -> Any:
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self.qubit_map, self.final_density_matrix.tolist())

    def __str__(self) -> str:
        samples = super().__str__()
        ret = f'measurements: {samples}'
        for substate in self._get_substates():
            tensor = substate.target_tensor
            size = np.prod([tensor.shape[i] for i in range(tensor.ndim // 2)], dtype=np.int64)
            dm = tensor.reshape((size, size))
            label = f'qubits: {substate.qubits}' if substate.qubits else 'phase:'
            ret += f'\n\n{label}\nfinal density matrix:\n{dm}'
        return ret

    def __repr__(self) -> str:
        return f'cirq.DensityMatrixTrialResult(params={self.params!r}, measurements={proper_repr(self.measurements)}, final_simulator_state={self._final_simulator_state!r})'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text('cirq.DensityMatrixTrialResult(...)' if cycle else self.__str__())