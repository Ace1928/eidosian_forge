import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
class LPIFilter2D:
    """Linear Position-Invariant Filter (2-dimensional)"""

    def __init__(self, impulse_response, **filter_params):
        """
        Parameters
        ----------
        impulse_response : callable `f(r, c, **filter_params)`
            Function that yields the impulse response.  ``r`` and ``c`` are
            1-dimensional vectors that represent row and column positions, in
            other words coordinates are (r[0],c[0]),(r[0],c[1]) etc.
            `**filter_params` are passed through.

            In other words, ``impulse_response`` would be called like this:

            >>> def impulse_response(r, c, **filter_params):
            ...     pass
            >>>
            >>> r = [0,0,0,1,1,1,2,2,2]
            >>> c = [0,1,2,0,1,2,0,1,2]
            >>> filter_params = {'kw1': 1, 'kw2': 2, 'kw3': 3}
            >>> impulse_response(r, c, **filter_params)


        Examples
        --------
        Gaussian filter without normalization of coefficients:

        >>> def filt_func(r, c, sigma=1):
        ...     return np.exp(-(r**2 + c**2)/(2 * sigma**2))
        >>> filter = LPIFilter2D(filt_func)

        """
        if not callable(impulse_response):
            raise ValueError('Impulse response must be a callable.')
        self.impulse_response = impulse_response
        self.filter_params = filter_params
        self._cache = None

    def _prepare(self, data):
        """Calculate filter and data FFT in preparation for filtering."""
        dshape = np.array(data.shape)
        even_offset = (dshape % 2 == 0).astype(int)
        dshape += even_offset
        oshape = np.array(data.shape) * 2 - 1
        float_dtype = _supported_float_type(data.dtype)
        data = data.astype(float_dtype, copy=False)
        if self._cache is None or np.any(self._cache.shape != oshape):
            coords = np.mgrid[[slice(0 + offset, float(n + offset)) for n, offset in zip(dshape, even_offset)]]
            for k, coord in enumerate(coords):
                coord -= (dshape[k] - 1) / 2.0
            coords = coords.reshape(2, -1).T
            coords = coords.astype(float_dtype, copy=False)
            f = self.impulse_response(coords[:, 0], coords[:, 1], **self.filter_params).reshape(dshape)
            f = _pad(f, oshape)
            F = fft.fftn(f)
            self._cache = F
        else:
            F = self._cache
        data = _pad(data, oshape)
        G = fft.fftn(data)
        return (F, G)

    def __call__(self, data):
        """Apply the filter to the given data.

        Parameters
        ----------
        data : (M, N) ndarray

        """
        check_nD(data, 2, 'data')
        F, G = self._prepare(data)
        out = fft.ifftn(F * G)
        out = np.abs(_center(out, data.shape))
        return out