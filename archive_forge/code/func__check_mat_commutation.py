import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def _check_mat_commutation(op1, op2):
    """Uses matrices and matrix multiplication to determine whether op1 and op2 commute.

    ``op1`` and ``op2`` must be on the same wires.
    """
    op1_mat = op1.matrix()
    op2_mat = op2.matrix()
    mat_12 = np.matmul(op1_mat, op2_mat)
    mat_21 = np.matmul(op2_mat, op1_mat)
    return qml.math.allclose(mat_12, mat_21)