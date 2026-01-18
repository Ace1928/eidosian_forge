import numpy as np
from scipy.sparse.linalg import aslinearoperator
@_blocked_elementwise
def _max_abs_axis1(X):
    return np.max(np.abs(X), axis=1)