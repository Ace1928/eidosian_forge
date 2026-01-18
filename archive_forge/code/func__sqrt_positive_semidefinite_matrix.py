from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def _sqrt_positive_semidefinite_matrix(mat: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite matrix."""
    eigs, vecs = linalg.eigh(mat)
    return vecs @ (np.sqrt(np.abs(eigs)) * vecs).T.conj()