import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, shortest_path
from ..base import (
from ..decomposition import KernelPCA
from ..metrics.pairwise import _VALID_METRICS
from ..neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from ..preprocessing import KernelCenterer
from ..utils._param_validation import Interval, StrOptions
from ..utils.graph import _fix_connected_components
from ..utils.validation import check_is_fitted
Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).

        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)
            X transformed in the new space.
        