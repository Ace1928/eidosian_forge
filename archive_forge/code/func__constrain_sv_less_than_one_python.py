import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _constrain_sv_less_than_one_python(unconstrained, order=None, k_endog=None):
    """
    Transform arbitrary matrices to matrices with singular values less than
    one.

    Parameters
    ----------
    unconstrained : list
        Arbitrary matrices. Should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    constrained : list
        Partial autocorrelation matrices. Should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`.

    See Also
    --------
    constrain_stationary_multivariate

    Notes
    -----
    Corresponds to Lemma 2.2 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """
    from scipy import linalg
    constrained = []
    if order is None:
        order = len(unconstrained)
    if k_endog is None:
        k_endog = unconstrained[0].shape[0]
    eye = np.eye(k_endog)
    for i in range(order):
        A = unconstrained[i]
        B, lower = linalg.cho_factor(eye + np.dot(A, A.T), lower=True)
        constrained.append(linalg.solve_triangular(B, A, lower=lower))
    return constrained