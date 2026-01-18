import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (zc) and real (zr)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    scipy.signal.cmplxreal

    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """
    z = cupy.atleast_1d(z)
    if z.size == 0:
        return (z, z)
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')
    if tol is None:
        tol = 100 * cupy.finfo((1.0 * z).dtype).eps
    z = z[cupy.lexsort(cupy.array([abs(z.imag), z.real]))]
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real
    if len(zr) == len(z):
        return (cupy.array([]), zr)
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]
    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching conjugate.')
    same_real = cupy.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = cupy.diff(cupy.r_[0, same_real, 0])
    run_starts = cupy.nonzero(diffs > 0)[0]
    run_stops = cupy.nonzero(diffs < 0)[0]
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[cupy.lexsort(cupy.array([abs(chunk.imag)]))]
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching conjugate.')
    zc = (zp + zn.conj()) / 2
    return (zc, zr)