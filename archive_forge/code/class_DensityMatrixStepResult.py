from typing import Any, Dict, List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import ops, protocols, study, value
from cirq._compat import proper_repr
from cirq.sim import simulator, density_matrix_simulation_state, simulator_base
class DensityMatrixStepResult(simulator_base.StepResultBase['cirq.DensityMatrixSimulationState']):
    """A single step in the simulation of the DensityMatrixSimulator.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(self, sim_state: 'cirq.SimulationStateBase[cirq.DensityMatrixSimulationState]', dtype: Type[np.complexfloating]=np.complex64):
        """DensityMatrixStepResult.

        Args:
            sim_state: The qubit:SimulationState lookup for this step.
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
        """
        super().__init__(sim_state)
        self._dtype = dtype
        self._density_matrix: Optional[np.ndarray] = None

    def density_matrix(self, copy=True):
        """Returns the density matrix at this step in the simulation.

        The density matrix that is stored in this result is returned in the
        computational basis with these basis states defined by the qubit_map.
        In particular the value in the qubit_map is the index of the qubit,
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

        Args:
            copy: If True, then the returned state is a copy of the density
                matrix. If False, then the density matrix is not copied,
                potentially saving memory. If one only needs to read derived
                parameters from the density matrix and store then using False
                can speed up simulation by eliminating a memory copy.
        """
        if self._density_matrix is None:
            self._density_matrix = np.array(1)
            state = self._merged_sim_state
            if state is not None:
                matrix = state.target_tensor
                size = int(np.sqrt(np.prod(matrix.shape, dtype=np.int64)))
                self._density_matrix = np.reshape(matrix, (size, size))
        return self._density_matrix.copy() if copy else self._density_matrix

    def __repr__(self) -> str:
        return f'cirq.DensityMatrixStepResult(sim_state={self._sim_state!r}, dtype=np.{np.dtype(self._dtype)!r})'