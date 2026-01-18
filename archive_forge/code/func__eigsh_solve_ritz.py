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
def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    alpha = cupy.asnumpy(alpha)
    beta = cupy.asnumpy(beta)
    t = numpy.diag(alpha)
    t = t + numpy.diag(beta[:-1], k=1)
    t = t + numpy.diag(beta[:-1], k=-1)
    if beta_k is not None:
        beta_k = cupy.asnumpy(beta_k)
        t[k, :k] = beta_k
        t[:k, k] = beta_k
    w, s = numpy.linalg.eigh(t)
    if which == 'LA':
        idx = numpy.argsort(w)
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'LM':
        idx = numpy.argsort(numpy.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'SA':
        idx = numpy.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]
    return (cupy.array(wk), cupy.array(sk))