from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
def _pauli_decompose(matrix, num_wires):
    """Compute the coefficients of a matrix or a batch of matrices (batch dimension(s) in the
    leading axes) in the Pauli basis.

    Args:
        matrix (tensor_like): Matrix or batch of matrices to decompose into the Pauli basis.
        num_wires (int): Number of wires the matrices act on.

    Returns:
        tensor_like: Coefficients of the input ``matrix`` in the Pauli basis.

    For a matrix :math:`M`, these coefficients are defined via
    :math:`M = \\sum_\\ell c_\\ell P_\\ell` and they can be computed using the (Frobenius) inner
    product of :math:`M` with the corresponding Pauli word :math:`P_\\ell`:
    :math:`c_\\ell = \\frac{1}{2^N}\\operatorname{Tr}\\left\\{P_\\ell M\\right\\}` where the prefactor
    is the normalization that makes the standard Pauli basis orthonormal, and :math:`N`
    is the number of qubits.
    That is, the normalization is such that a single Pauli word :math:`P_k` has
    coefficients ``c_\\ell = \\delta_{k\\ell}``.

    Note that this implementation takes :math:`\\mathcal{O}(16^N)` operations per input
    matrix but there is a more efficient method taking only :math:`\\mathcal{O}(N4^N)`
    operations per matrix.
    """
    basis = pauli_basis_matrices(num_wires)
    coefficients = qml.math.tensordot(basis, matrix, axes=[[1, 2], [-1, -2]])
    return qml.math.cast(coefficients, matrix.dtype) / 2 ** num_wires