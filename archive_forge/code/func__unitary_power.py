from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _unitary_power(matrix: np.ndarray, power: float) -> np.ndarray:
    return map_eigenvalues(matrix, lambda e: e ** power)