import warnings
from typing import Any, List, Sequence, Optional
import numpy as np
from cirq import devices, linalg, ops, protocols
from cirq.testing import lin_alg_utils
def _reorder_indices_of_matrix(matrix: np.ndarray, new_order: List[int]):
    num_qubits = matrix.shape[0].bit_length() - 1
    matrix = np.reshape(matrix, (2,) * 2 * num_qubits)
    all_indices = range(2 * num_qubits)
    new_input_indices = new_order
    new_output_indices = [i + num_qubits for i in new_input_indices]
    matrix = np.moveaxis(matrix, all_indices, new_input_indices + new_output_indices)
    matrix = np.reshape(matrix, (2 ** num_qubits, 2 ** num_qubits))
    return matrix