from ..libmp.backend import xrange
from .calculus import defun

    Evaluates a Fourier series (in the format computed by
    by :func:`~mpmath.fourier` for the given interval) at the point `x`.

    The series should be a pair `(c, s)` where `c` is the
    cosine series and `s` is the sine series. The two lists
    need not have the same length.
    