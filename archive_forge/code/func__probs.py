from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
def _probs(density_matrix: np.ndarray, indices: Sequence[int], qid_shape: Tuple[int, ...]) -> np.ndarray:
    """Returns the probabilities for a measurement on the given indices."""
    all_probs = np.diagonal(np.reshape(density_matrix, (np.prod(qid_shape, dtype=np.int64),) * 2))
    return simulation_utils.state_probabilities_by_indices(all_probs.real, indices, qid_shape)