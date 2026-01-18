import math
from collections.abc import Iterable
from warnings import warn
import numpy as np
from numpy import random
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
from .._shared import utils
from .._shared.filters import gaussian
from ..color import rgb2lab
from ..util import img_as_float, regular_grid
from ._slic import _enforce_label_connectivity_cython, _slic_cython
def _get_mask_centroids(mask, n_centroids, multichannel):
    """Find regularly spaced centroids on a mask.

    Parameters
    ----------
    mask : 3D ndarray
        The mask within which the centroids must be positioned.
    n_centroids : int
        The number of centroids to be returned.

    Returns
    -------
    centroids : 2D ndarray
        The coordinates of the centroids with shape (n_centroids, 3).
    steps : 1D ndarray
        The approximate distance between two seeds in all dimensions.

    """
    coord = np.array(np.nonzero(mask), dtype=float).T
    rng = random.RandomState(123)
    idx_full = np.arange(len(coord), dtype=int)
    idx = np.sort(rng.choice(idx_full, min(n_centroids, len(coord)), replace=False))
    dense_factor = 10
    ndim_spatial = mask.ndim - 1 if multichannel else mask.ndim
    n_dense = int(dense_factor ** ndim_spatial * n_centroids)
    if len(coord) > n_dense:
        idx_dense = np.sort(rng.choice(idx_full, n_dense, replace=False))
    else:
        idx_dense = Ellipsis
    centroids, _ = kmeans2(coord[idx_dense], coord[idx], iter=5)
    dist = squareform(pdist(centroids))
    np.fill_diagonal(dist, np.inf)
    closest_pts = dist.argmin(-1)
    steps = abs(centroids - centroids[closest_pts, :]).mean(0)
    return (centroids, steps)