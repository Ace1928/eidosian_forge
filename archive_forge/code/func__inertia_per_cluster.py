import warnings
import numpy as np
import scipy.sparse as sp
from ..base import _fit_context
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Integral, Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from ._k_means_common import _inertia_dense, _inertia_sparse
from ._kmeans import (
def _inertia_per_cluster(self, X, centers, labels, sample_weight):
    """Calculate the sum of squared errors (inertia) per cluster.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The input samples.

        centers : ndarray of shape (n_clusters=2, n_features)
            The cluster centers.

        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        Returns
        -------
        inertia_per_cluster : ndarray of shape (n_clusters=2,)
            Sum of squared errors (inertia) for each cluster.
        """
    n_clusters = centers.shape[0]
    _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
    inertia_per_cluster = np.empty(n_clusters)
    for label in range(n_clusters):
        inertia_per_cluster[label] = _inertia(X, sample_weight, centers, labels, self._n_threads, single_label=label)
    return inertia_per_cluster