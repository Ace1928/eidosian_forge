from numbers import Integral, Real
from time import time
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from ..base import (
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative
from . import _barnes_hut_tsne, _utils  # type: ignore
def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components, angle=0.5, skip_num_points=0, verbose=False, compute_error=True, num_threads=1):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)
    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)
    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr, grad, angle, n_components, verbose, dof=degrees_of_freedom, compute_error=compute_error, num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c
    return (error, grad)