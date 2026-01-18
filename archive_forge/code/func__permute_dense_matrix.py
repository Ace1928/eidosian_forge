import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def _permute_dense_matrix(matrix, wires, wire_order, batch_dim):
    """Permute the matrix to match the wires given in `wire_order`.

    Args:
        matrix (np.ndarray): matrix to permute
        wires (list): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (list): global wire order, which has to contain all wire labels in ``wires``,
            but can also contain additional labels
        batch_dim (int or None): Batch dimension. If ``None``, batching is ignored.

    Returns:
        np.ndarray: permuted matrix
    """
    if wires == wire_order:
        return matrix
    perm = [wires.index(wire) for wire in wire_order]
    num_wires = len(wire_order)
    perm += [p + num_wires for p in perm]
    if batch_dim:
        perm = [0] + [p + 1 for p in perm]
    shape = [batch_dim] + [2] * (num_wires * 2) if batch_dim else [2] * (num_wires * 2)
    matrix = qml.math.reshape(matrix, shape)
    matrix = qml.math.transpose(matrix, axes=perm)
    shape = [batch_dim] + [2 ** num_wires] * 2 if batch_dim else [2 ** num_wires] * 2
    return qml.math.reshape(matrix, shape)