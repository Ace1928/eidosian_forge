import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _expectation_from_state_vector_no_validation(self, state_vector: np.ndarray, qubit_map: Mapping[TKey, int]) -> float:
    """Evaluate the expectation of this PauliString given a state vector.

        This method does not provide input validation. See
        `PauliString.expectation_from_state_vector` for function description.

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
    if len(state_vector.shape) == 1:
        num_qubits = state_vector.shape[0].bit_length() - 1
        state_vector = np.reshape(state_vector, (2,) * num_qubits)
    ket = np.copy(state_vector)
    for qubit, pauli in self.items():
        buffer = np.empty(ket.shape, dtype=state_vector.dtype)
        args = protocols.ApplyUnitaryArgs(target_tensor=ket, available_buffer=buffer, axes=(qubit_map[qubit],))
        ket = protocols.apply_unitary(pauli, args)
    return self.coefficient * np.tensordot(state_vector.conj(), ket, axes=len(ket.shape)).item()