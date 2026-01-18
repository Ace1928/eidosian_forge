import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy._core import internal
from cupy.cuda import device
from cupyx.cusolver import check_availability
from cupyx.cusolver import _gesvdj_batched, _gesvd_batched
from cupyx.cusolver import _geqrf_orgqr_batched
from cupy.linalg import _util
Singular Value Decomposition.

    Factorizes the matrix ``a`` as ``u * np.diag(s) * v``, where ``u`` and
    ``v`` are unitary and ``s`` is an one-dimensional array of ``a``'s
    singular values.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(..., M, N)``.
        full_matrices (bool): If True, it returns u and v with dimensions
            ``(..., M, M)`` and ``(..., N, N)``. Otherwise, the dimensions
            of u and v are ``(..., M, K)`` and ``(..., K, N)``, respectively,
            where ``K = min(M, N)``.
        compute_uv (bool): If ``False``, it only returns singular values.

    Returns:
        tuple of :class:`cupy.ndarray`:
            A tuple of ``(u, s, v)`` such that ``a = u * np.diag(s) * v``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. note::
        On CUDA, when ``a.ndim > 2`` and the matrix dimensions <= 32, a fast
        code path based on Jacobian method (``gesvdj``) is taken. Otherwise,
        a QR method (``gesvd``) is used.

        On ROCm, there is no such a fast code path that switches the underlying
        algorithm.

    .. seealso:: :func:`numpy.linalg.svd`
    