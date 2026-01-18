import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def _canonicalize_kak_vector(k_vec: np.ndarray, atol: float) -> np.ndarray:
    """Map a KAK vector into its Weyl chamber equivalent vector.

    This implementation is vectorized but does not produce the single qubit
    unitaries required to bring the KAK vector into canonical form.

    Args:
        k_vec: THe KAK vector to be canonicalized. This input may be vectorized,
            with shape (...,3), where the final axis denotes the k_vector and
            all other axes are broadcast.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0.

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0
        The output is vectorized, with shape k_vec.shape[:-1] + (3,).
    """
    k_vec = np.mod(k_vec + np.pi / 4, np.pi / 2) - np.pi / 4
    order = np.argsort(np.abs(k_vec), axis=-1)
    k_vec = np.take_along_axis(k_vec, order, axis=-1)[..., ::-1]
    x_negative = k_vec[..., 0] < 0
    k_vec[x_negative, 0] *= -1
    k_vec[x_negative, 2] *= -1
    y_negative = k_vec[..., 1] < 0
    k_vec[y_negative, 1] *= -1
    k_vec[y_negative, 2] *= -1
    x_is_pi_over_4 = np.isclose(k_vec[..., 0], np.pi / 4, atol=atol)
    z_is_negative = k_vec[..., 2] < 0
    need_diff = np.logical_and(x_is_pi_over_4, z_is_negative)
    k_vec[need_diff, 2] *= -1
    return k_vec