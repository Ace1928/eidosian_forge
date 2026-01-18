import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import (
from ._k_means_common import (
from ._k_means_elkan import (
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse
def _init_centroids(self, X, x_squared_norms, init, random_state, sample_weight, init_size=None, n_centroids=None):
    """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape                 (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
    n_samples = X.shape[0]
    n_clusters = self.n_clusters if n_centroids is None else n_centroids
    if init_size is not None and init_size < n_samples:
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
        sample_weight = sample_weight[init_indices]
    if isinstance(init, str) and init == 'k-means++':
        centers, _ = _kmeans_plusplus(X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms, sample_weight=sample_weight)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.choice(n_samples, size=n_clusters, replace=False, p=sample_weight / sample_weight.sum())
        centers = X[seeds]
    elif _is_arraylike_not_scalar(self.init):
        centers = init
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = check_array(centers, dtype=X.dtype, copy=False, order='C')
        self._validate_center_shape(X, centers)
    if sp.issparse(centers):
        centers = centers.toarray()
    return centers