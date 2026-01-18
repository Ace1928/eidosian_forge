from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
Solve an equation system, ``a * x = b``, given the LU factorization of ``a``

    Args:
        lu_and_piv (tuple): LU factorization of matrix ``a`` (``(M, M)``)
            together with pivot indices.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        trans ({0, 1, 2}): Type of system to solve:

            ========  =========
            trans     system
            ========  =========
            0         a x  = b
            1         a^T x = b
            2         a^H x = b
            ========  =========
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.lu_solve`
    