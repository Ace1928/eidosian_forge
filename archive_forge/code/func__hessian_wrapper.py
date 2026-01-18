import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common
def _hessian_wrapper(hess, x, z, k_params):
    """
    Wraps the hessian up in the form for cvxopt.

    cvxopt wants the hessian of the objective function and the constraints.
        Since our constraints are linear, this part is all zeros.
    """
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    zh_x = np.asarray(z[0]) * hess(params)
    zero_mat = np.zeros(zh_x.shape)
    A = np.concatenate((zh_x, zero_mat), axis=1)
    B = np.concatenate((zero_mat, zero_mat), axis=1)
    zh_x_ext = np.concatenate((A, B), axis=0)
    return matrix(zh_x_ext, (2 * k_params, 2 * k_params))