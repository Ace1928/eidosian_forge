from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _decomp_3sqrt_iswap_matrices(kak: 'cirq.KakDecomposition', atol: float=1e-08) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 3-SQRT_ISWAP decomposition.

    Assumes any canonical x, y, z.  Three sqrt-iSWAP gates are only needed if
    x < y + |z|.  Only two are needed for other gates (most cases).

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = kak.interaction_coefficients
    b0, b1 = kak.single_qubit_operations_before
    a0, a1 = kak.single_qubit_operations_after
    ieq1 = y > np.pi / 8
    ieq2 = z < 0
    if ieq1:
        if ieq2:
            x1, y1, z1 = (0.0, np.pi / 8, -np.pi / 8)
        else:
            x1, y1, z1 = (0.0, np.pi / 8, np.pi / 8)
    else:
        x1, y1, z1 = (-np.pi / 8, np.pi / 8, 0.0)
    x2, y2, z2 = (x - x1, y - y1, z - z1)
    kak1 = linalg.kak_canonicalize_vector(x1, y1, z1, atol)
    kak2 = linalg.kak_canonicalize_vector(x2, y2, z2, atol)
    ((h0, h1), (g0, g1)), phase1 = _decomp_1sqrt_iswap_matrices(kak1, atol)
    ((e0, e1), (c0, c1), (d0, d1)), phase2 = _decomp_2sqrt_iswap_matrices(kak2, atol)
    return ([(h0 @ b0, h1 @ b1), (e0 @ g0, e1 @ g1), (c0, c1), (a0 @ d0, a1 @ d1)], kak.global_phase * phase1 * phase2)