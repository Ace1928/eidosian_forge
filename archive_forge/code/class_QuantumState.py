from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
class QuantumState:
    """A quantum state.

    Can be a state vector, a state tensor, or a density matrix.
    """

    def __init__(self, data: np.ndarray, qid_shape: Optional[Tuple[int, ...]]=None, *, dtype: Optional['DTypeLike']=None, validate: bool=True, atol: float=1e-07) -> None:
        """Initialize a quantum state object.

        Args:
            data: The data representing the quantum state.
            qid_shape: The qid shape.
            validate: Whether to check if the given data and qid shape
                represent a valid quantum state with the given dtype.
            dtype: The expected data type of the quantum state.
            atol: Absolute numerical tolerance to use for validation.

        Raises:
            ValueError: The qid shape was not specified and could not be
                inferred.
            ValueError: Invalid quantum state.
        """
        if qid_shape is None:
            qid_shape = infer_qid_shape(data)
        self._data = data
        self._qid_shape = qid_shape
        self._dim = np.prod(self.qid_shape, dtype=np.int64).item()
        if validate:
            self.validate(dtype=dtype, atol=atol)

    @property
    def data(self) -> np.ndarray:
        """The data underlying the quantum state."""
        return self._data

    @property
    def qid_shape(self) -> Tuple[int, ...]:
        """The qid shape of the quantum state."""
        return self._qid_shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the quantum state."""
        return self._data.dtype

    def state_vector(self) -> Optional[np.ndarray]:
        """Return the state vector of this state.

        A state vector stores the amplitudes of a pure state as a
        one-dimensional array.
        If the state is a density matrix, this method returns None.
        """
        if self._is_density_matrix():
            return None
        return np.reshape(self.data, (self._dim,))

    def state_tensor(self) -> Optional[np.ndarray]:
        """Return the state tensor of this state.

        A state tensor stores the amplitudes of a pure state as an array with
        shape equal to the qid shape of the state.
        If the state is a density matrix, this method returns None.
        """
        if self._is_density_matrix():
            return None
        return np.reshape(self.data, self.qid_shape)

    def density_matrix(self) -> np.ndarray:
        """Return the density matrix of this state.

        A density matrix stores the entries of a density matrix as a matrix
        (a two-dimensional array).
        """
        if not self._is_density_matrix():
            state_vector = self.state_vector()
            assert state_vector is not None, 'only None if _is_density_matrix'
            return np.outer(state_vector, np.conj(state_vector))
        return self.data

    def state_vector_or_density_matrix(self) -> np.ndarray:
        """Return the state vector or density matrix of this state.

        If the state is a denity matrix, return the density matrix. Otherwise, return the state
        vector.
        """
        state_vector = self.state_vector()
        if state_vector is not None:
            return state_vector
        return self.data

    def _is_density_matrix(self) -> bool:
        """Whether this quantum state is a density matrix."""
        return self.data.shape == (self._dim, self._dim)

    def validate(self, *, dtype: Optional['DTypeLike']=None, atol=1e-07) -> None:
        """Check if this quantum state is valid.

        Args:
            dtype: The expected data type of the quantum state.
            atol: Absolute numerical tolerance to use for validation.

        Raises:
            ValueError: Invalid quantum state.
        """
        is_state_vector = self.data.shape == (self._dim,)
        is_state_tensor = self.data.shape == self.qid_shape
        if is_state_vector or is_state_tensor:
            state_vector = self.state_vector()
            assert state_vector is not None
            validate_normalized_state_vector(state_vector, qid_shape=self.qid_shape, dtype=dtype, atol=atol)
        elif self._is_density_matrix():
            validate_density_matrix(self.density_matrix(), qid_shape=self.qid_shape, dtype=dtype, atol=atol)
        else:
            raise ValueError(f'Invalid quantum state: Data shape of {self.data.shape} is not compatible with qid shape of {self.qid_shape}.')