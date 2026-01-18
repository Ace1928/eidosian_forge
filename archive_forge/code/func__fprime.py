import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common
def _fprime(score, x, k_params, alpha):
    """
    The regularized derivative.
    """
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    fprime_arr = np.append(score(params), alpha)
    return matrix(fprime_arr, (1, 2 * k_params))