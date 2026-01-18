from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
def _sample_qmallows(n, rng=None):
    """Sample from the quantum Mallows distribution"""
    if rng is None:
        rng = np.random.default_rng()
    had = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=int)
    inds = list(range(n))
    for i in range(n):
        m = n - i
        eps = 4 ** (-m)
        r = rng.uniform(0, 1)
        index = -int(np.ceil(np.log2(r + (1 - r) * eps)))
        had[i] = index < m
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        perm[i] = inds[k]
        del inds[k]
    return (had, perm)