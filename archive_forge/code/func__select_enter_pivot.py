import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _select_enter_pivot(c_hat, bl, a, rule='bland', tol=1e-12):
    """
    Selects a pivot to enter the basis. Currently Bland's rule - the smallest
    index that has a negative reduced cost - is the default.
    """
    if rule.lower() == 'mrc':
        return a[~bl][np.argmin(c_hat)]
    else:
        return a[~bl][c_hat < -tol][0]