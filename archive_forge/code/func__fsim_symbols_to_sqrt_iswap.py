from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _fsim_symbols_to_sqrt_iswap(a: 'cirq.Qid', b: 'cirq.Qid', theta: 'cirq.TParamVal', phi: 'cirq.TParamVal', use_sqrt_iswap_inv: bool=True):
    """Implements `cirq.FSimGate(theta, phi)(a, b)` using two √iSWAPs and single qubit rotations.

    FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π)

    Args:
        a: The first qubit.
        b: The second qubit.
        theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
        phi: Controlled phase angle, in radians.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    if theta != 0.0:
        yield _iswap_symbols_to_sqrt_iswap(a, b, -2 * theta / np.pi, use_sqrt_iswap_inv)
    if phi != 0.0:
        yield _cphase_symbols_to_sqrt_iswap(a, b, -phi / np.pi, use_sqrt_iswap_inv)