import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _su2su2_to_tensor_products(U):
    """Given a matrix :math:`U = A \\otimes B` in SU(2) x SU(2), extract the two SU(2)
    operations A and B.

    This process has been described in detail in the Appendix of Coffey & Deiotte
    https://link.springer.com/article/10.1007/s11128-009-0156-3
    """
    C1 = U[0:2, 0:2]
    C2 = U[0:2, 2:4]
    C3 = U[2:4, 0:2]
    C4 = U[2:4, 2:4]
    C14 = math.dot(C1, math.conj(math.T(C4)))
    a1 = math.sqrt(math.cast_like(C14[0, 0], 1j))
    C23 = math.dot(C2, math.conj(math.T(C3)))
    a2 = math.sqrt(-math.cast_like(C23[0, 0], 1j))
    C12 = math.dot(C1, math.conj(math.T(C2)))
    if not math.is_abstract(C12):
        if not math.allclose(a1 * math.conj(a2), C12[0, 0]):
            a2 *= -1
    else:
        sign_is_correct = math.allclose(a1 * math.conj(a2), C12[0, 0])
        sign = (-1) ** (sign_is_correct + 1)
        a2 *= sign
    A = math.stack([math.stack([a1, a2]), math.stack([-math.conj(a2), math.conj(a1)])])
    use_B2 = math.allclose(A[0, 0], 0.0, atol=1e-06)
    if not math.is_abstract(A):
        B = C2 / math.cast_like(A[0, 1], 1j) if use_B2 else C1 / math.cast_like(A[0, 0], 1j)
    elif qml.math.get_interface(A) == 'jax':
        B = qml.math.cond(use_B2, lambda x: C2 / math.cast_like(A[0, 1], 1j), lambda x: C1 / math.cast_like(A[0, 0], 1j), [0])
    return (math.convert_like(A, U), math.convert_like(B, U))