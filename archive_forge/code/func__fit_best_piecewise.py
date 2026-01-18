from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse
from scipy.sparse.linalg import eigsh, svds
from ..base import BaseEstimator, BiclusterMixin, _fit_context
from ..utils import check_random_state, check_scalar
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import make_nonnegative, randomized_svd, safe_sparse_dot
from ..utils.validation import assert_all_finite
from ._kmeans import KMeans, MiniBatchKMeans
def _fit_best_piecewise(self, vectors, n_best, n_clusters):
    """Find the ``n_best`` vectors that are best approximated by piecewise
        constant vectors.

        The piecewise vectors are found by k-means; the best is chosen
        according to Euclidean distance.

        """

    def make_piecewise(v):
        centroid, labels = self._k_means(v.reshape(-1, 1), n_clusters)
        return centroid[labels].ravel()
    piecewise_vectors = np.apply_along_axis(make_piecewise, axis=1, arr=vectors)
    dists = np.apply_along_axis(norm, axis=1, arr=vectors - piecewise_vectors)
    result = vectors[np.argsort(dists)[:n_best]]
    return result