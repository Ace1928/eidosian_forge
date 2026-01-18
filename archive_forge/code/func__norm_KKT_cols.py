from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def _norm_KKT_cols(self, P, A):
    """
        Compute the norm of the KKT matrix from P and A
        """
    norm_P_cols = spspa.linalg.norm(P, np.inf, axis=0)
    norm_A_cols = spspa.linalg.norm(A, np.inf, axis=0)
    norm_first_half = np.maximum(norm_P_cols, norm_A_cols)
    norm_second_half = spspa.linalg.norm(A, np.inf, axis=1)
    return np.hstack((norm_first_half, norm_second_half))