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
def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(distances_data, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), 'All probabilities should be finite'
    P = csr_matrix((conditional_P.ravel(), distances.indices, distances.indptr), shape=(n_samples, n_samples))
    P = P + P.T
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P
    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print('[t-SNE] Computed conditional probabilities in {:.3f}s'.format(duration))
    return P