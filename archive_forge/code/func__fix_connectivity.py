import warnings
from heapq import heapify, heappop, heappush, heappushpop
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from ..base import (
from ..metrics import DistanceMetric
from ..metrics._dist_metrics import METRIC_MAPPING64
from ..metrics.pairwise import _VALID_METRICS, paired_distances
from ..utils import check_array
from ..utils._fast_dict import IntFloatDict
from ..utils._param_validation import (
from ..utils.graph import _fix_connected_components
from ..utils.validation import check_memory
from . import _hierarchical_fast as _hierarchical  # type: ignore
from ._feature_agglomeration import AgglomerationTransform
def _fix_connectivity(X, connectivity, affinity):
    """
    Fixes the connectivity matrix.

    The different steps are:

    - copies it
    - makes it symmetric
    - converts it to LIL if necessary
    - completes it if necessary.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    affinity : {"euclidean", "precomputed"}, default="euclidean"
        Which affinity to use. At the moment `precomputed` and
        ``euclidean`` are supported. `euclidean` uses the
        negative squared Euclidean distance between points.

    Returns
    -------
    connectivity : sparse matrix
        The fixed connectivity matrix.

    n_connected_components : int
        The number of connected components in the graph.
    """
    n_samples = X.shape[0]
    if connectivity.shape[0] != n_samples or connectivity.shape[1] != n_samples:
        raise ValueError('Wrong shape for connectivity matrix: %s when X is %s' % (connectivity.shape, X.shape))
    connectivity = connectivity + connectivity.T
    if not sparse.issparse(connectivity):
        connectivity = sparse.lil_matrix(connectivity)
    if connectivity.format != 'lil':
        connectivity = connectivity.tolil()
    n_connected_components, labels = connected_components(connectivity)
    if n_connected_components > 1:
        warnings.warn('the number of connected components of the connectivity matrix is %d > 1. Completing it to avoid stopping the tree early.' % n_connected_components, stacklevel=2)
        connectivity = _fix_connected_components(X=X, graph=connectivity, n_connected_components=n_connected_components, component_labels=labels, metric=affinity, mode='connectivity')
    return (connectivity, n_connected_components)