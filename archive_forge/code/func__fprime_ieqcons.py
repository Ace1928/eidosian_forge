import numpy as np
from scipy.optimize import fmin_slsqp
import statsmodels.base.l1_solvers_common as l1_solvers_common
def _fprime_ieqcons(x_full, k_params):
    """
    Derivative of the inequality constraints
    """
    I = np.eye(k_params)
    A = np.concatenate((I, I), axis=1)
    B = np.concatenate((-I, I), axis=1)
    C = np.concatenate((A, B), axis=0)
    return C