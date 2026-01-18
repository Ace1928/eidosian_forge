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
def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    if not sparse.issparse(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            laplacian = laplacian.tocsr()
    return laplacian