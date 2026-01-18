import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def _local_invariants_from_kak(vector: np.ndarray) -> np.ndarray:
    """Local invariants of a two-qubit unitary from its KAK vector.

    Any 2 qubit unitary may be expressed as

    $U = k_l A k_r$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$
    A = \\exp( i * \\sum_{j=x,y,z} k_j \\sigma_{(j,0)}\\sigma_{(j,1)})
    $$

    Here $(k_x,k_y,k_z)$ is the KAK vector.

    Args:
        vector: Shape (...,3) tensor representing different KAK vectors.

    Returns:
        The local invariants associated with the given KAK vector. Shape
        (..., 3), where first two elements are the real and imaginary parts
        of G1 and the third is G2.

    References:
        "A geometric theory of non-local two-qubit operations"
        https://arxiv.org/abs/quant-ph/0209120
    """
    vector = np.asarray(vector)
    kx = vector[..., 0]
    ky = vector[..., 1]
    kz = vector[..., 2]
    cos, sin = (np.cos, np.sin)
    G1R = (cos(2 * kx) * cos(2 * ky) * cos(2 * kz)) ** 2
    G1R -= (sin(2 * kx) * sin(2 * ky) * sin(2 * kz)) ** 2
    G1I = 0.25 * sin(4 * kx) * sin(4 * ky) * sin(4 * kz)
    G2 = cos(4 * kx) + cos(4 * ky) + cos(4 * kz)
    return np.moveaxis(np.array([G1R, G1I, G2]), 0, -1)