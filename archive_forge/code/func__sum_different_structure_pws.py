import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
@staticmethod
def _sum_different_structure_pws(indices, data):
    """Sums Pauli words with different parse structures."""
    size = indices.shape[0]
    idx = np.argsort(indices, axis=1)
    matrix = sparse.csr_matrix((size, size), dtype='complex128')
    matrix.indices = np.take_along_axis(indices, idx, axis=1).ravel()
    matrix.data = np.take_along_axis(data, idx, axis=1).ravel()
    num_entries_per_row = indices.shape[1]
    matrix.indptr = _cached_arange(size + 1) * num_entries_per_row
    matrix.data[np.abs(matrix.data) < 1e-16] = 0
    matrix.eliminate_zeros()
    return matrix