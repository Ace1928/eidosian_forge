import cmath
import numbers
import cupy
from numpy import pi
from cupyx.scipy.fft import fft, ifft, next_fast_len
class CZT:
    """
    Create a callable chirp z-transform function.

    Transform to compute the frequency response around a spiral.
    Objects of this class are callables which can compute the
    chirp z-transform on their inputs.  This object precalculates the constant
    chirps used in the given transform.

    Parameters
    ----------
    n : int
        The size of the signal.
    m : int, optional
        The number of output points desired.  Default is `n`.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    f : CZT
        Callable object ``f(x, axis=-1)`` for computing the chirp z-transform
        on `x`.

    See Also
    --------
    czt : Convenience function for quickly calculating CZT.
    ZoomFFT : Class that creates a callable partial FFT function.
    scipy.signal.CZT

    Notes
    -----
    The defaults are chosen such that ``f(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, m)`` is equivalent to
    ``fft.fft(x, m)``.

    If `w` does not lie on the unit circle, then the transform will be
    around a spiral with exponentially-increasing radius.  Regardless,
    angle will increase linearly.

    For transforms that do lie on the unit circle, accuracy is better when
    using `ZoomFFT`, since any numerical error in `w` is
    accumulated for long data lengths, drifting away from the unit circle.

    The chirp z-transform can be faster than an equivalent FFT with
    zero padding.  Try it with your own array sizes to see.

    However, the chirp z-transform is considerably less precise than the
    equivalent zero-padded FFT.

    As this CZT is implemented using the Bluestein algorithm [1]_, it can
    compute large prime-length Fourier transforms in O(N log N) time, rather
    than the O(N**2) time required by the direct DFT calculation.
    (`scipy.fft` also uses Bluestein's algorithm'.)

    (The name "chirp z-transform" comes from the use of a chirp in the
    Bluestein algorithm [2]_.  It does not decompose signals into chirps, like
    other transforms with "chirp" in the name.)

    References
    ----------
    .. [1] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).
    .. [2] Rabiner, Schafer, and Rader, "The chirp z-transform algorithm and
           its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    """

    def __init__(self, n, m=None, w=None, a=1 + 0j):
        m = _validate_sizes(n, m)
        k = cupy.arange(max(m, n), dtype=cupy.min_scalar_type(-max(m, n) ** 2))
        if w is None:
            w = cmath.exp(-2j * pi / m)
            wk2 = cupy.exp(-(1j * pi * (k ** 2 % (2 * m))) / m)
        else:
            wk2 = w ** (k ** 2 / 2.0)
        a = 1.0 * a
        self.w, self.a = (w, a)
        self.m, self.n = (m, n)
        nfft = next_fast_len(n + m - 1)
        self._Awk2 = a ** (-k[:n]) * wk2[:n]
        self._nfft = nfft
        self._Fwk2 = fft(1 / cupy.hstack((wk2[n - 1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n - 1, n + m - 1)

    def __call__(self, x, *, axis=-1):
        """
        Calculate the chirp z-transform of a signal.

        Parameters
        ----------
        x : array
            The signal to transform.
        axis : int, optional
            Axis over which to compute the FFT. If not given, the last axis is
            used.

        Returns
        -------
        out : ndarray
            An array of the same dimensions as `x`, but with the length of the
            transformed axis set to `m`.
        """
        x = cupy.asarray(x)
        if x.shape[axis] != self.n:
            raise ValueError(f'CZT defined for length {self.n}, not {x.shape[axis]}')
        trnsp = list(range(x.ndim))
        trnsp[axis], trnsp[-1] = (trnsp[-1], trnsp[axis])
        x = x.transpose(*trnsp)
        y = ifft(self._Fwk2 * fft(x * self._Awk2, self._nfft))
        y = y[..., self._yidx] * self._wk2
        return y.transpose(*trnsp)

    def points(self):
        """
        Return the points at which the chirp z-transform is computed.
        """
        return czt_points(self.m, self.w, self.a)