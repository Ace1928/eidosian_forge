import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def _sparse_swap_mat(qubit_i, qubit_j, n):
    """Helper function which generates the sparse matrix of SWAP
    for qubits: i <--> j with final shape (2**n, 2**n)."""

    def swap_qubits(index, i, j):
        s = list(format(index, f'0{n}b'))
        si, sj = (s[i], s[j])
        if si == sj:
            return index
        s[i], s[j] = (sj, si)
        return int(f'0b{''.join(s)}', 2)
    data = [1] * 2 ** n
    index_i = list(range(2 ** n))
    index_j = [swap_qubits(idx, qubit_i, qubit_j) for idx in index_i]
    return csr_matrix((data, (index_i, index_j)))