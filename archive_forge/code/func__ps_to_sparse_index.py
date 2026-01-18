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
def _ps_to_sparse_index(pauli_words, wires):
    """Represent the Pauli words sparse structure in a matrix of shape n_words x n_wires."""
    indices = np.zeros((len(pauli_words), len(wires)))
    for i, pw in enumerate(pauli_words):
        if not pw.wires:
            continue
        wire_indices = np.array(wires.indices(pw.wires))
        indices[i, wire_indices] = [pauli_to_sparse_int[pw[w]] for w in pw.wires]
    return indices