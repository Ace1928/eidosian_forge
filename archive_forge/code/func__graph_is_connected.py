import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import (
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.fixes import parse_version, sp_version
def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    if sparse.issparse(graph):
        accept_large_sparse = sp_version >= parse_version('1.11.3')
        graph = check_array(graph, accept_sparse=True, accept_large_sparse=accept_large_sparse)
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]