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
def _random_cd(best_sample: np.ndarray, n_iters: int, n_nochange: int, rng: GeneratorType, **kwargs: dict) -> np.ndarray:
    """Optimal LHS on CD.

    Create a base LHS and do random permutations of coordinates to
    lower the centered discrepancy.
    Because it starts with a normal LHS, it also works with the
    `scramble` keyword argument.

    Two stopping criterion are used to stop the algorithm: at most,
    `n_iters` iterations are performed; or if there is no improvement
    for `n_nochange` consecutive iterations.
    """
    del kwargs
    n, d = best_sample.shape
    if d == 0 or n == 0:
        return np.empty((n, d))
    if d == 1 or n == 1:
        return best_sample
    best_disc = discrepancy(best_sample)
    bounds = ([0, d - 1], [0, n - 1], [0, n - 1])
    n_nochange_ = 0
    n_iters_ = 0
    while n_nochange_ < n_nochange and n_iters_ < n_iters:
        n_iters_ += 1
        col = rng_integers(rng, *bounds[0], endpoint=True)
        row_1 = rng_integers(rng, *bounds[1], endpoint=True)
        row_2 = rng_integers(rng, *bounds[2], endpoint=True)
        disc = _perturb_discrepancy(best_sample, row_1, row_2, col, best_disc)
        if disc < best_disc:
            best_sample[row_1, col], best_sample[row_2, col] = (best_sample[row_2, col], best_sample[row_1, col])
            best_disc = disc
            n_nochange_ = 0
        else:
            n_nochange_ += 1
    return best_sample