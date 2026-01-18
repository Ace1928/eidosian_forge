import numpy
import cupy
import cupyx.scipy.sparse
def _sparse_frobenius_norm(x):
    if cupy.issubdtype(x.dtype, cupy.complexfloating):
        sqnorm = abs(x).power(2).sum()
    else:
        sqnorm = x.power(2).sum()
    return cupy.sqrt(sqnorm)