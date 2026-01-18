import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._disjoint_set import DisjointSet
def _randdm(pnts):
    """
    Generate a random distance matrix stored in condensed form.

    Parameters
    ----------
    pnts : int
        The number of points in the distance matrix. Has to be at least 2.

    Returns
    -------
    D : ndarray
        A ``pnts * (pnts - 1) / 2`` sized vector is returned.
    """
    if pnts >= 2:
        D = np.random.rand(pnts * (pnts - 1) / 2)
    else:
        raise ValueError('The number of points in the distance matrix must be at least 2.')
    return D