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
def _get_same_structure_csr(self, pauli_words, wire_order):
    """Returns the CSR indices and data for Pauli words with the same sparse structure."""
    indices = pauli_words[0]._get_csr_indices(wire_order)
    nwires = len(wire_order)
    nwords = len(pauli_words)
    inner = np.empty((nwords, 2 ** (nwires - nwires // 2)), dtype=np.complex128)
    outer = np.empty((nwords, 2 ** (nwires // 2)), dtype=np.complex128)
    for i, word in enumerate(pauli_words):
        outer[i, :], inner[i, :] = word._get_csr_data_2(wire_order, coeff=qml.math.to_numpy(self[word]))
    data = outer.T @ inner
    return (indices, data.ravel())