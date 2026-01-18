from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
def _van_der_corput_permutations(base: IntNumber, *, random_state: SeedType=None) -> np.ndarray:
    """Permutations for scrambling a Van der Corput sequence.

    Parameters
    ----------
    base : int
        Base of the sequence.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Returns
    -------
    permutations : array_like
        Permutation indices.

    Notes
    -----
    In Algorithm 1 of Owen 2017, a permutation of `np.arange(base)` is
    created for each positive integer `k` such that `1 - base**-k < 1`
    using floating-point arithmetic. For double precision floats, the
    condition `1 - base**-k < 1` can also be written as `base**-k >
    2**-54`, which makes it more apparent how many permutations we need
    to create.
    """
    rng = check_random_state(random_state)
    count = math.ceil(54 / math.log2(base)) - 1
    permutations = np.repeat(np.arange(base)[None], count, axis=0)
    for perm in permutations:
        rng.shuffle(perm)
    return permutations