from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _in_1sqrt_iswap_region(interaction_coefficients: Tuple[float, float, float], weyl_tol: float=1e-08) -> bool:
    """Tests if (x, y, z) ~= (π/8, π/8, 0), assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x - np.pi / 8) <= weyl_tol and abs(y - np.pi / 8) <= weyl_tol and (abs(z) <= weyl_tol)