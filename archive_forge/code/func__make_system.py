import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def _make_system(A, M, x0, b):
    """Make a linear system Ax = b

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): sparse or dense matrix.
        M (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): preconditioner.
        x0 (cupy.ndarray): initial guess to iterative method.
        b (cupy.ndarray): right hand side.

    Returns:
        tuple:
            It returns (A, M, x, b).
            A (LinaerOperator): matrix of linear system
            M (LinearOperator): preconditioner
            x (cupy.ndarray): initial guess
            b (cupy.ndarray): right hand side.
    """
    fast_matvec = _make_fast_matvec(A)
    A = _interface.aslinearoperator(A)
    if fast_matvec is not None:
        A = _interface.LinearOperator(A.shape, matvec=fast_matvec, rmatvec=A.rmatvec, dtype=A.dtype)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(A.dtype))
    n = A.shape[0]
    if not (b.shape == (n,) or b.shape == (n, 1)):
        raise ValueError('b has incompatible dimensions')
    b = b.astype(A.dtype).ravel()
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0.astype(A.dtype).ravel()
    if M is None:
        M = _interface.IdentityOperator(shape=A.shape, dtype=A.dtype)
    else:
        fast_matvec = _make_fast_matvec(M)
        M = _interface.aslinearoperator(M)
        if fast_matvec is not None:
            M = _interface.LinearOperator(M.shape, matvec=fast_matvec, rmatvec=M.rmatvec, dtype=M.dtype)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')
    return (A, M, x, b)