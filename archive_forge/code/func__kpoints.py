from k-means models and quantizing vectors by comparing them with
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
from scipy._lib._util import check_random_state, rng_integers
from scipy.spatial.distance import cdist
from . import _vq
def _kpoints(data, k, rng, xp):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    idx = rng.choice(data.shape[0], size=int(k), replace=False)
    return data[idx, ...]