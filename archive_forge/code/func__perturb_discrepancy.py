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
def _perturb_discrepancy(sample: np.ndarray, i1: int, i2: int, k: int, disc: float):
    """Centered discrepancy after an elementary perturbation of a LHS.

    An elementary perturbation consists of an exchange of coordinates between
    two points: ``sample[i1, k] <-> sample[i2, k]``. By construction,
    this operation conserves the LHS properties.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample (before permutation) to compute the discrepancy from.
    i1 : int
        The first line of the elementary permutation.
    i2 : int
        The second line of the elementary permutation.
    k : int
        The column of the elementary permutation.
    disc : float
        Centered discrepancy of the design before permutation.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the design after permutation.

    References
    ----------
    .. [1] Jin et al. "An efficient algorithm for constructing optimal design
       of computer experiments", Journal of Statistical Planning and
       Inference, 2005.

    """
    n = sample.shape[0]
    z_ij = sample - 0.5
    c_i1j = 1.0 / n ** 2.0 * np.prod(0.5 * (2.0 + abs(z_ij[i1, :]) + abs(z_ij) - abs(z_ij[i1, :] - z_ij)), axis=1)
    c_i2j = 1.0 / n ** 2.0 * np.prod(0.5 * (2.0 + abs(z_ij[i2, :]) + abs(z_ij) - abs(z_ij[i2, :] - z_ij)), axis=1)
    c_i1i1 = 1.0 / n ** 2 * np.prod(1 + abs(z_ij[i1, :])) - 2.0 / n * np.prod(1.0 + 0.5 * abs(z_ij[i1, :]) - 0.5 * z_ij[i1, :] ** 2)
    c_i2i2 = 1.0 / n ** 2 * np.prod(1 + abs(z_ij[i2, :])) - 2.0 / n * np.prod(1.0 + 0.5 * abs(z_ij[i2, :]) - 0.5 * z_ij[i2, :] ** 2)
    num = 2 + abs(z_ij[i2, k]) + abs(z_ij[:, k]) - abs(z_ij[i2, k] - z_ij[:, k])
    denum = 2 + abs(z_ij[i1, k]) + abs(z_ij[:, k]) - abs(z_ij[i1, k] - z_ij[:, k])
    gamma = num / denum
    c_p_i1j = gamma * c_i1j
    c_p_i2j = c_i2j / gamma
    alpha = (1 + abs(z_ij[i2, k])) / (1 + abs(z_ij[i1, k]))
    beta = (2 - abs(z_ij[i2, k])) / (2 - abs(z_ij[i1, k]))
    g_i1 = np.prod(1.0 + abs(z_ij[i1, :]))
    g_i2 = np.prod(1.0 + abs(z_ij[i2, :]))
    h_i1 = np.prod(1.0 + 0.5 * abs(z_ij[i1, :]) - 0.5 * z_ij[i1, :] ** 2)
    h_i2 = np.prod(1.0 + 0.5 * abs(z_ij[i2, :]) - 0.5 * z_ij[i2, :] ** 2)
    c_p_i1i1 = g_i1 * alpha / n ** 2 - 2.0 * alpha * beta * h_i1 / n
    c_p_i2i2 = g_i2 / (n ** 2 * alpha) - 2.0 * h_i2 / (n * alpha * beta)
    sum_ = c_p_i1j - c_i1j + c_p_i2j - c_i2j
    mask = np.ones(n, dtype=bool)
    mask[[i1, i2]] = False
    sum_ = sum(sum_[mask])
    disc_ep = disc + c_p_i1i1 - c_i1i1 + c_p_i2i2 - c_i2i2 + 2 * sum_
    return disc_ep