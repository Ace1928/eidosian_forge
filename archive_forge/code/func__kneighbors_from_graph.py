import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def _kneighbors_from_graph(graph, n_neighbors, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    n_neighbors : int
        Number of neighbors required for each sample.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples, n_neighbors)
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples, n_neighbors)
        Indices of nearest neighbors.
    """
    n_samples = graph.shape[0]
    assert graph.format == 'csr'
    row_nnz = np.diff(graph.indptr)
    row_nnz_min = row_nnz.min()
    if n_neighbors is not None and row_nnz_min < n_neighbors:
        raise ValueError('%d neighbors per samples are required, but some samples have only %d neighbors in precomputed graph matrix. Decrease number of neighbors used or recompute the graph with more neighbors.' % (n_neighbors, row_nnz_min))

    def extract(a):
        if row_nnz.max() == row_nnz_min:
            return a.reshape(n_samples, -1)[:, :n_neighbors]
        else:
            idx = np.tile(np.arange(n_neighbors), (n_samples, 1))
            idx += graph.indptr[:-1, None]
            return a.take(idx, mode='clip').reshape(n_samples, n_neighbors)
    if return_distance:
        return (extract(graph.data), extract(graph.indices))
    else:
        return extract(graph.indices)