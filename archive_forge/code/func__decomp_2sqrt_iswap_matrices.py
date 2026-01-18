from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _decomp_2sqrt_iswap_matrices(kak: 'cirq.KakDecomposition', atol: float=1e-08) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 2-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and x >= y + |z| within tolerance.  For x, y, z
    that violate this inequality, three sqrt-iSWAP gates are required.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = kak.interaction_coefficients
    b0, b1 = kak.single_qubit_operations_before
    a0, a1 = kak.single_qubit_operations_after

    def safe_arccos(v):
        return np.arccos(np.clip(v, -1, 1))

    def nonzero_sign(v):
        return -1 if v < 0 else 1
    _c = np.clip(np.sin(x + y - z) * np.sin(x - y + z) * np.sin(-x - y - z) * np.sin(-x + y + z), 0, 1)
    alpha = safe_arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) + 2 * np.sqrt(_c))
    beta = safe_arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) - 2 * np.sqrt(_c))
    _4ccs = 4 * (np.cos(x) * np.cos(z) * np.sin(y)) ** 2
    gamma = safe_arccos(nonzero_sign(z) * np.sqrt(_4ccs / (_4ccs + np.clip(np.cos(2 * x) * np.cos(2 * y) * np.cos(2 * z), 0, 1))))
    c0 = protocols.unitary(ops.rz(-gamma)) @ protocols.unitary(ops.rx(-alpha)) @ protocols.unitary(ops.rz(-gamma))
    c1 = protocols.unitary(ops.rx(-beta))
    u_sqrt_iswap = protocols.unitary(ops.SQRT_ISWAP)
    u = u_sqrt_iswap @ np.kron(c0, c1) @ u_sqrt_iswap
    kak_fix = linalg.kak_decomposition(u, atol=atol / 10, rtol=0, check_preconditions=False)
    e0, e1 = kak_fix.single_qubit_operations_before
    d0, d1 = kak_fix.single_qubit_operations_after
    return ([(e0.T.conj() @ b0, e1.T.conj() @ b1), (c0, c1), (a0 @ d0.T.conj(), a1 @ d1.T.conj())], kak.global_phase / kak_fix.global_phase)