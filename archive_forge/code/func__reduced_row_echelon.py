import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def _reduced_row_echelon(binary_matrix):
    """Returns the reduced row echelon form (RREF) of a matrix in a binary finite field :math:`\\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix representation of the Hamiltonian
    Returns:
        array[int]: reduced row-echelon form of the given `binary_matrix`

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                           [1, 0, 1, 0, 0, 0, 1, 0],
    ...                           [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> _reduced_row_echelon(binary_matrix)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    rref_mat = binary_matrix.copy()
    shape = rref_mat.shape
    icol = 0
    for irow in range(shape[0]):
        while icol < shape[1] and (not rref_mat[irow][icol]):
            non_zero_idx = rref_mat[irow:, icol].nonzero()[0]
            if len(non_zero_idx) == 0:
                icol += 1
            else:
                krow = irow + non_zero_idx[0]
                rref_mat[irow, icol:], rref_mat[krow, icol:] = (rref_mat[krow, icol:].copy(), rref_mat[irow, icol:].copy())
        if icol < shape[1] and rref_mat[irow][icol]:
            rpvt_cols = rref_mat[irow, icol:].copy()
            currcol = rref_mat[:, icol].copy()
            currcol[irow] = 0
            rref_mat[:, icol:] ^= np.outer(currcol, rpvt_cols)
            icol += 1
    return rref_mat.astype(int)