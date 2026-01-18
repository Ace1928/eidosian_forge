import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
Computes the roots of a polynomial with given coefficients.

    Args:
        p (cupy.ndarray or cupy.poly1d): polynomial coefficients.

    Returns:
        cupy.ndarray: polynomial roots.

    .. warning::

        This function doesn't support currently polynomial coefficients
        whose companion matrices are general 2d square arrays. Only those
        with complex Hermitian or real symmetric 2d arrays are allowed.

        The current `cupy.roots` doesn't guarantee the order of results.

    .. seealso:: :func:`numpy.roots`

    