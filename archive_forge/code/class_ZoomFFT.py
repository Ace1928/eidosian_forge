import cmath
import numbers
import cupy
from numpy import pi
from cupyx.scipy.fft import fft, ifft, next_fast_len
class ZoomFFT(CZT):
    """
    Create a callable zoom FFT transform function.

    This is a specialization of the chirp z-transform (`CZT`) for a set of
    equally-spaced frequencies around the unit circle, used to calculate a
    section of the FFT more efficiently than calculating the entire FFT and
    truncating. [1]_

    Parameters
    ----------
    n : int
        The size of the signal.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  Default is `n`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    f : ZoomFFT
        Callable object ``f(x, axis=-1)`` for computing the zoom FFT on `x`.

    See Also
    --------
    zoom_fft : Convenience function for calculating a zoom FFT.
    scipy.signal.ZoomFFT

    Notes
    -----
    The defaults are chosen such that ``f(x, 2)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, 2, m)`` is equivalent to
    ``fft.fft(x, m)``.

    Sampling frequency is 1/dt, the time step between samples in the
    signal `x`.  The unit circle corresponds to frequencies from 0 up
    to the sampling frequency.  The default sampling frequency of 2
    means that `f1`, `f2` values up to the Nyquist frequency are in the
    range [0, 1). For `f1`, `f2` values expressed in radians, a sampling
    frequency of 2*pi should be used.

    Remember that a zoom FFT can only interpolate the points of the existing
    FFT.  It cannot help to resolve two separate nearby frequencies.
    Frequency resolution can only be increased by increasing acquisition
    time.

    These functions are implemented using Bluestein's algorithm (as is
    `scipy.fft`). [2]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 29 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf
    .. [2] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).
    """

    def __init__(self, n, fn, m=None, *, fs=2, endpoint=False):
        m = _validate_sizes(n, m)
        k = cupy.arange(max(m, n), dtype=cupy.min_scalar_type(-max(m, n) ** 2))
        fn = cupy.asarray(fn)
        if cupy.size(fn) == 2:
            f1, f2 = fn
        elif cupy.size(fn) == 1:
            f1, f2 = (0.0, fn)
        else:
            raise ValueError('fn must be a scalar or 2-length sequence')
        self.f1, self.f2, self.fs = (f1, f2, fs)
        if endpoint:
            scale = (f2 - f1) * m / (fs * (m - 1))
        else:
            scale = (f2 - f1) / fs
        a = cmath.exp(2j * pi * f1 / fs)
        wk2 = cupy.exp(-(1j * pi * scale * k ** 2) / m)
        self.w = cmath.exp(-2j * pi / m * scale)
        self.a = a
        self.m, self.n = (m, n)
        ak = cupy.exp(-2j * pi * f1 / fs * k[:n])
        self._Awk2 = ak * wk2[:n]
        nfft = next_fast_len(n + m - 1)
        self._nfft = nfft
        self._Fwk2 = fft(1 / cupy.hstack((wk2[n - 1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n - 1, n + m - 1)