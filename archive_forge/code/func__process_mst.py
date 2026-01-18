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
def _process_mst(min_spanning_tree):
    """
    Builds a single-linkage tree (SLT) from the provided minimum spanning tree
    (MST). The MST is first sorted then processed by a custom Cython routine.

    Parameters
    ----------
    min_spanning_tree : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    row_order = np.argsort(min_spanning_tree['distance'])
    min_spanning_tree = min_spanning_tree[row_order]
    return make_single_linkage(min_spanning_tree)