from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied

    Shift periodic sequence x by a: y(u) = x(u+a).

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

          y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a : float
        Defines the parameters of the sinh/sinh pseudo-differential
    period : float, optional
        The period of the sequences x and y. Default period is ``2*pi``.
    