from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _bipartite_tensor(mat1, mat2, shape1=None, shape2=None):
    """Tensor product (A ⊗ B) to bipartite matrices and reravel indices.

    This is used for tensor product of superoperators and Choi matrices.

    Args:
        mat1 (matrix_like): a bipartite matrix A
        mat2 (matrix_like): a bipartite matrix B
        shape1 (tuple): bipartite-shape for matrix A (a0, a1, a2, a3)
        shape2 (tuple): bipartite-shape for matrix B (b0, b1, b2, b3)

    Returns:
        np.array: a bipartite matrix for reravel(A ⊗ B).

    Raises:
        QiskitError: if input matrices are wrong shape.
    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    dim_a0, dim_a1 = mat1.shape
    dim_b0, dim_b1 = mat2.shape
    if shape1 is None:
        sdim_a0 = int(np.sqrt(dim_a0))
        sdim_a1 = int(np.sqrt(dim_a1))
        shape1 = (sdim_a0, sdim_a0, sdim_a1, sdim_a1)
    if shape2 is None:
        sdim_b0 = int(np.sqrt(dim_b0))
        sdim_b1 = int(np.sqrt(dim_b1))
        shape2 = (sdim_b0, sdim_b0, sdim_b1, sdim_b1)
    if len(shape1) != 4 or shape1[0] * shape1[1] != dim_a0 or shape1[2] * shape1[3] != dim_a1:
        raise QiskitError('Invalid shape_a')
    if len(shape2) != 4 or shape2[0] * shape2[1] != dim_b0 or shape2[2] * shape2[3] != dim_b1:
        raise QiskitError('Invalid shape_b')
    return _reravel(mat1, mat2, shape1, shape2)