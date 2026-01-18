import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def _augmented_orthnormal_cols(x, n_aug):
    if n_aug <= 0:
        return x
    m, n = x.shape
    y = cupy.empty((m, n + n_aug), dtype=x.dtype)
    y[:, :n] = x
    for i in range(n, n + n_aug):
        v = cupy.random.random((m,)).astype(x.dtype)
        v -= v @ y[:, :i].conj() @ y[:, :i].T
        y[:, i] = v / cupy.linalg.norm(v)
    return y