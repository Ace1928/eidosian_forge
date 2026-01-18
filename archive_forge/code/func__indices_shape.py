from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
def _indices_shape(qid_shape: Tuple[int, ...], indices: Sequence[int]) -> Tuple[int, ...]:
    """Validates that the indices have values within range of `len(qid_shape)`."""
    if any((index < 0 for index in indices)):
        raise IndexError(f'Negative index in indices: {indices}')
    if any((index >= len(qid_shape) for index in indices)):
        raise IndexError(f'Out of range indices, must be less than number of qubits but was {indices}')
    return tuple((qid_shape[i] for i in indices))