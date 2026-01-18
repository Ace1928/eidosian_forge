import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _dct_or_dst_type2(x, n=None, axis=-1, forward=True, norm=None, dst=False, overwrite_x=False):
    """Forward DCT/DST-II (or inverse DCT/DST-III) along a single axis

    Parameters
    ----------
    x : cupy.ndarray
        The data to transform.
    n : int
        The size of the transform. If None, ``x.shape[axis]`` is used.
    axis : int
        Axis along which the transform is applied.
    forward : bool
        Set true to indicate that this is a forward DCT-II as opposed to an
        inverse DCT-III (The difference between the two is only in the
        normalization factor).
    norm : {None, 'ortho', 'forward', 'backward'}
        The normalization convention to use.
    dst : bool
        If True, a discrete sine transform is computed rather than the discrete
        cosine transform.
    overwrite_x : bool
        Indicates that it is okay to overwrite x. In practice, the current
        implementation never performs the transform in-place.

    Returns
    -------
    y: cupy.ndarray
        The transformed array.
    """
    if axis < -x.ndim or axis >= x.ndim:
        raise numpy.AxisError('axis out of range')
    if axis < 0:
        axis += x.ndim
    if n is not None and n < 1:
        raise ValueError(f'invalid number of data points ({n}) specified')
    x = _cook_shape(x, (n,), (axis,), 'R2R')
    n = x.shape[axis]
    x = _reshuffle_dct2(x, x.shape[axis], axis, dst)
    if norm == 'ortho':
        inorm = 'sqrt'
    elif norm == 'forward':
        inorm = 'full' if forward else 'none'
    else:
        inorm = 'none' if forward else 'full'
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=2)
    x = _fft.fft(x, n=n, axis=axis, overwrite_x=True)
    tmp = _exp_factor_dct2(x, n, axis, norm_factor)
    x *= tmp
    x = cupy.real(x)
    if dst:
        slrev = [slice(None)] * x.ndim
        slrev[axis] = slice(None, None, -1)
        x = x[tuple(slrev)]
    if norm == 'ortho':
        sl0 = [slice(None)] * x.ndim
        sl0[axis] = slice(1)
        x[tuple(sl0)] *= math.sqrt(2) * 0.5
    return x