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
def _lanczos_asis(a, V, u, alpha, beta, i_start, i_end):
    for i in range(i_start, i_end):
        u[...] = a @ V[i]
        cublas.dotc(V[i], u, out=alpha[i])
        u -= u.T @ V[:i + 1].conj().T @ V[:i + 1]
        cublas.nrm2(u, out=beta[i])
        if i >= i_end - 1:
            break
        V[i + 1] = u / beta[i]