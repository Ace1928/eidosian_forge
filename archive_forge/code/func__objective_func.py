import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common
def _objective_func(f, x, k_params, alpha, *args):
    """
    The regularized objective function.
    """
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    u = x_arr[k_params:]
    objective_func_arr = f(params, *args) + (alpha * u).sum()
    return matrix(objective_func_arr)