import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.
    """
    if probability == 0:
        return 0
    if n_inliers == 0:
        return np.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    return np.ceil(np.log(nom) / np.log(denom))