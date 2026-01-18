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
def _get_csr_data(self, wire_order, coeff):
    """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
    full_word = [self[wire] for wire in wire_order]
    matrix_size = 2 ** len(wire_order)
    data = np.empty(matrix_size, dtype=np.complex128)
    current_size = 2
    data[:current_size], _ = _cached_sparse_data(full_word[-1])
    data[:current_size] *= coeff
    for s in full_word[-2::-1]:
        if s == 'I':
            data[current_size:2 * current_size] = data[:current_size]
        elif s == 'X':
            data[current_size:2 * current_size] = data[:current_size]
        elif s == 'Y':
            data[current_size:2 * current_size] = 1j * data[:current_size]
            data[:current_size] *= -1j
        elif s == 'Z':
            data[current_size:2 * current_size] = -data[:current_size]
        current_size *= 2
    return data