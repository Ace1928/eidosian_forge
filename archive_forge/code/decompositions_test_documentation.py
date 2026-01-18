import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
Local invariants of a two-qubit unitary from its KAK vector.

    Any 2 qubit unitary may be expressed as

    $U = k_l A k_r$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$
    A = \exp( i * \sum_{j=x,y,z} k_j \sigma_{(j,0)}\sigma_{(j,1)})
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
    