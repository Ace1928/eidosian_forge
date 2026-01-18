from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.random import default_rng
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.random import random_unitary
from .statevector import Statevector
from .densitymatrix import DensityMatrix
def _ginibre_matrix(nrow, ncol, seed):
    """Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed(int or np.random.Generator): default rng.

    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    ginibre = rng.normal(size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre