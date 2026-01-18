import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _unconstrain_sv_less_than_one(constrained, order=None, k_endog=None):
    """
    Transform matrices with singular values less than one to arbitrary
    matrices.

    Parameters
    ----------
    constrained : list
        The partial autocorrelation matrices. Should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    unconstrained : list
        Unconstrained matrices. A list of length `order`, where each element is
        an array sized `k_endog` x `k_endog`.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Corresponds to the inverse of Lemma 2.2 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.
    """
    from scipy import linalg
    unconstrained = []
    if order is None:
        order = len(constrained)
    if k_endog is None:
        k_endog = constrained[0].shape[0]
    eye = np.eye(k_endog)
    for i in range(order):
        P = constrained[i]
        B_inv, lower = linalg.cho_factor(eye - np.dot(P, P.T), lower=True)
        unconstrained.append(linalg.solve_triangular(B_inv, P, lower=lower))
    return unconstrained