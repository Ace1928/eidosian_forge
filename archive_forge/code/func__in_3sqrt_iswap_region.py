from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _in_3sqrt_iswap_region(interaction_coefficients: Tuple[float, float, float], weyl_tol: float=1e-08) -> bool:
    """Any two-qubit operation is decomposable into three SQRT_ISWAP gates.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    return True