import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplace_sym(m, d):
    return lambda v: v * d[:, np.newaxis] - m @ v - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))