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
class _BisectingTree:
    """Tree structure representing the hierarchical clusters of BisectingKMeans."""

    def __init__(self, center, indices, score):
        """Create a new cluster node in the tree.

        The node holds the center of this cluster and the indices of the data points
        that belong to it.
        """
        self.center = center
        self.indices = indices
        self.score = score
        self.left = None
        self.right = None

    def split(self, labels, centers, scores):
        """Split the cluster node into two subclusters."""
        self.left = _BisectingTree(indices=self.indices[labels == 0], center=centers[0], score=scores[0])
        self.right = _BisectingTree(indices=self.indices[labels == 1], center=centers[1], score=scores[1])
        self.indices = None

    def get_cluster_to_bisect(self):
        """Return the cluster node to bisect next.

        It's based on the score of the cluster, which can be either the number of
        data points assigned to that cluster or the inertia of that cluster
        (see `bisecting_strategy` for details).
        """
        max_score = None
        for cluster_leaf in self.iter_leaves():
            if max_score is None or cluster_leaf.score > max_score:
                max_score = cluster_leaf.score
                best_cluster_leaf = cluster_leaf
        return best_cluster_leaf

    def iter_leaves(self):
        """Iterate over all the cluster leaves in the tree."""
        if self.left is None:
            yield self
        else:
            yield from self.left.iter_leaves()
            yield from self.right.iter_leaves()