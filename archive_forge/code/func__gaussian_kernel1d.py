import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def _gaussian_kernel1d(sigma, order, radius, dtype=cupy.float64):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()
    if order == 0:
        return cupy.asarray(phi_x)
    exponent_range = numpy.arange(order + 1)
    q = numpy.zeros(order + 1)
    q[0] = 1
    D = numpy.diag(exponent_range[1:], 1)
    P = numpy.diag(numpy.ones(order) / -sigma2, -1)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1], dtype=dtype)