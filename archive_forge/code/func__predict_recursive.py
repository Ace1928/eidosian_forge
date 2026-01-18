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
def _predict_recursive(self, X, sample_weight, cluster_node):
    """Predict recursively by going down the hierarchical tree.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The data points, currently assigned to `cluster_node`, to predict between
            the subclusters of this node.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        cluster_node : _BisectingTree node object
            The cluster node of the hierarchical tree.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
    if cluster_node.left is None:
        return np.full(X.shape[0], cluster_node.label, dtype=np.int32)
    centers = np.vstack((cluster_node.left.center, cluster_node.right.center))
    if hasattr(self, '_X_mean'):
        centers += self._X_mean
    cluster_labels = _labels_inertia_threadpool_limit(X, sample_weight, centers, self._n_threads, return_inertia=False)
    mask = cluster_labels == 0
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels[mask] = self._predict_recursive(X[mask], sample_weight[mask], cluster_node.left)
    labels[~mask] = self._predict_recursive(X[~mask], sample_weight[~mask], cluster_node.right)
    return labels