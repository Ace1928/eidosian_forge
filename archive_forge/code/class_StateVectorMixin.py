import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils
class StateVectorMixin:
    """A mixin that provide methods for objects that have a state vector."""

    def __init__(self, qubit_map: Optional[Mapping['cirq.Qid', int]]=None, *args, **kwargs):
        """Inits StateVectorMixin.

        Args:
            qubit_map: A map from the Qubits in the Circuit to the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state (see the state_vector() method).
            *args: Passed on to the class that this is mixed in with.
            **kwargs: Passed on to the class that this is mixed in with.
        """
        super().__init__(*args, **kwargs)
        self._qubit_map = qubit_map or {}
        qid_shape = simulator._qubit_map_to_shape(self._qubit_map)
        self._qid_shape = None if qubit_map is None else qid_shape

    @property
    def qubit_map(self) -> Mapping['cirq.Qid', int]:
        return self._qubit_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        if self._qid_shape is None:
            return NotImplemented
        return self._qid_shape

    @abc.abstractmethod
    def state_vector(self, copy: bool=False) -> np.ndarray:
        """Return the state vector (wave function).

        The vector is returned in the computational basis with these basis
        states defined by the `qubit_map`. In particular the value in the
        `qubit_map` is the index of the qubit, and these are translated into
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
        raise NotImplementedError()

    def dirac_notation(self, decimals: int=2) -> str:
        """Returns the state vector as a string in Dirac notation.

        Args:
            decimals: How many decimals to include in the pretty print.

        Returns:
            A pretty string consisting of a sum of computational basis kets
            and non-zero floats of the specified accuracy."""
        return qis.dirac_notation(self.state_vector(), decimals, qid_shape=self._qid_shape)

    def density_matrix_of(self, qubits: Optional[List['cirq.Qid']]=None) -> np.ndarray:
        """Returns the density matrix of the state.

        Calculate the density matrix for the system on the qubits provided.
        Any qubits not in the list that are present in self.state_vector() will
        be traced out. If qubits is None, the full density matrix for
        self.state_vector() is returned, given self.state_vector() follows
        standard Kronecker convention of numpy.kron.

        For example, if `self.state_vector()` returns
        `np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)`,
        then `density_matrix_of(qubits = None)` gives us

        $$
        \\rho = \\begin{bmatrix}
                    0.5 & 0.5 \\\\
                    0.5 & 0.5
                \\end{bmatrix}
        $$

        Args:
            qubits: list containing qubit IDs that you would like
                to include in the density matrix (i.e.) qubits that WON'T
                be traced out.

        Returns:
            A numpy array representing the density matrix.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if the indices are out of range for the number of qubits
                corresponding to the state.
        """
        return qis.density_matrix_from_state_vector(self.state_vector(), [self.qubit_map[q] for q in qubits] if qubits is not None else None, qid_shape=self._qid_shape)

    def bloch_vector_of(self, qubit: 'cirq.Qid') -> np.ndarray:
        """Returns the bloch vector of a qubit in the state.

        Calculates the bloch vector of the given qubit
        in the state given by self.state_vector(), given that
        self.state_vector() follows the standard Kronecker convention of
        numpy.kron.

        Args:
            qubit: qubit who's bloch vector we want to find.

        Returns:
            A length 3 numpy array representing the qubit's bloch vector.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if index is out of range for the number of qubits
                corresponding to the state.
        """
        return qis.bloch_vector_from_state_vector(self.state_vector(), self.qubit_map[qubit], qid_shape=self._qid_shape)