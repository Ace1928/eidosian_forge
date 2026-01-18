import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        lower (bool): Use only data contained in the lower triangle of ``a``.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T' or 'C'): Type of system to solve:

            - *'0'* or *'N'* -- :math:`a x  = b`
            - *'1'* or *'T'* -- :math:`a^T x = b`
            - *'2'* or *'C'* -- :math:`a^H x = b`

        unit_diagonal (bool): If ``True``, diagonal elements of ``a`` are
            assumed to be 1 and will not be referenced.
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.solve_triangular`
    