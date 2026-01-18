from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable
from .finite_difference import _processing_fn, finite_diff_coeffs
from .gradient_transform import (
from .general_shift_rules import generate_multishifted_tapes
def _rademacher_sampler(indices, num_params, *args, rng):
    """Sample a random vector with (independent) entries from {+1, -1} with balanced probability.
    That is, each entry follows the
    `Rademacher distribution. <https://en.wikipedia.org/wiki/Rademacher_distribution>`_

    The signature corresponds to the one required for the input ``sampler`` to ``spsa_grad``:

    Args:
        indices (Sequence[int]): Indices of the trainable tape parameters that will be perturbed.
        num_params (int): Total number of trainable tape parameters.
        rng (np.random.Generator): A NumPy pseudo-random number generator.

    Returns:
        tensor_like: Vector of size ``num_params`` with non-zero entries at positions indicated
        by ``indices``, each entry sampled independently from the Rademacher distribution.
    """
    direction = np.zeros(num_params)
    direction[indices] = rng.choice([-1, 1], size=len(indices))
    return direction