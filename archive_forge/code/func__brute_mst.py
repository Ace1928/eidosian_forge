from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import csgraph, issparse
from ...base import BaseEstimator, ClusterMixin, _fit_context
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import (
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels
def _brute_mst(mutual_reachability, min_samples):
    """
    Builds a minimum spanning tree (MST) from the provided mutual-reachability
    values. This function dispatches to a custom Cython implementation for
    dense arrays, and `scipy.sparse.csgraph.minimum_spanning_tree` for sparse
    arrays/matrices.

    Parameters
    ----------
    mututal_reachability_graph: {ndarray, sparse matrix} of shape             (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Returns
    -------
    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.
    """
    if not issparse(mutual_reachability):
        return mst_from_mutual_reachability(mutual_reachability)
    indptr = mutual_reachability.indptr
    num_points = mutual_reachability.shape[0]
    if any((indptr[i + 1] - indptr[i] < min_samples for i in range(num_points))):
        raise ValueError(f'There exists points with fewer than {min_samples} neighbors. Ensure your distance matrix has non-zero values for at least `min_sample`={min_samples} neighbors for each points (i.e. K-nn graph), or specify a `max_distance` in `metric_params` to use when distances are missing.')
    n_components = csgraph.connected_components(mutual_reachability, directed=False, return_labels=False)
    if n_components > 1:
        raise ValueError(f'Sparse mutual reachability matrix has {n_components} connected components. HDBSCAN cannot be perfomed on a disconnected graph. Ensure that the sparse distance matrix has only one connected component.')
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability)
    rows, cols = sparse_min_spanning_tree.nonzero()
    mst = np.rec.fromarrays([rows, cols, sparse_min_spanning_tree.data], dtype=MST_edge_dtype)
    return mst