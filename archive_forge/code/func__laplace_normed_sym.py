import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplace_normed_sym(m, d, nd):
    laplace_sym = _laplace_sym(m, d)
    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])