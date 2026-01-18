from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _sqrt_iswap_inv(a: 'cirq.Qid', b: 'cirq.Qid', use_sqrt_iswap_inv: bool=True) -> 'cirq.OP_TREE':
    """Optree implementing `cirq.SQRT_ISWAP_INV(a, b)` using √iSWAPs.

    Args:
        a: The first qubit.
        b: The second qubit.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Returns:
        `cirq.SQRT_ISWAP_INV(a, b)` or equivalent unitary implemented using `cirq.SQRT_ISWAP`.
    """
    return ops.SQRT_ISWAP_INV(a, b) if use_sqrt_iswap_inv else [ops.Z(a), ops.SQRT_ISWAP(a, b), ops.Z(a)]