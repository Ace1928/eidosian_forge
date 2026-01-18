from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma
from ..metrics.cluster import mutual_info_score
from ..neighbors import KDTree, NearestNeighbors
from ..preprocessing import scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_array, check_X_y
def _compute_mi_cd(c, d, n_neighbors):
    """Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    n_samples = c.shape[0]
    c = c.reshape((-1, 1))
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]
    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all)
    mi = digamma(n_samples) + np.mean(digamma(k_all)) - np.mean(digamma(label_counts)) - np.mean(digamma(m_all))
    return max(0, mi)