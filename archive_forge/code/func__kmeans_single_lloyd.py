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
def _kmeans_single_lloyd(X, sample_weight, centers_init, max_iter=300, verbose=False, tol=0.0001, n_threads=1):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense
    strict_convergence = False
    with threadpool_limits(limits=1, user_api='blas'):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, centers, centers_new, weight_in_clusters, labels, center_shift, n_threads)
            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f'Iteration {i}, inertia {inertia}.')
            centers, centers_new = (centers_new, centers)
            if np.array_equal(labels, labels_old):
                if verbose:
                    print(f'Converged at iteration {i}: strict convergence.')
                strict_convergence = True
                break
            else:
                center_shift_tot = (center_shift ** 2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(f'Converged at iteration {i}: center shift {center_shift_tot} within tolerance {tol}.')
                    break
            labels_old[:] = labels
        if not strict_convergence:
            lloyd_iter(X, sample_weight, centers, centers, weight_in_clusters, labels, center_shift, n_threads, update_centers=False)
    inertia = _inertia(X, sample_weight, centers, labels, n_threads)
    return (labels, inertia, centers, i + 1)