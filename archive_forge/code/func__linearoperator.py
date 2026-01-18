import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _linearoperator(mv, shape, dtype):
    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)