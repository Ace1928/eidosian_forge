import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def _deconstruct_single_qubit_matrix_into_gate_turns(mat: np.ndarray) -> Tuple[float, float, float]:
    """Breaks down a 2x2 unitary into gate parameters.

    Args:
        mat: The 2x2 unitary matrix to break down.

    Returns:
       A tuple containing the amount to rotate around an XY axis, the phase of
       that axis, and the amount to phase around Z. All results will be in
       fractions of a whole turn, with values canonicalized into the range
       [-0.5, 0.5).
    """
    pre_phase, rotation, post_phase = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
    tau = 2 * np.pi
    xy_turn = rotation / tau
    xy_phase_turn = 0.25 - pre_phase / tau
    total_z_turn = (post_phase + pre_phase) / tau
    return (_signed_mod_1(xy_turn), _signed_mod_1(xy_phase_turn), _signed_mod_1(total_z_turn))