import math
import cupy
from cupy.linalg import _util
Compute the matrix exponential.

    Parameters
    ----------
    a : ndarray, 2D

    Returns
    -------
    matrix exponential of `a`

    Notes
    -----
    Uses (a simplified) version of Algorithm 2.3 of [1]_:
    a [13 / 13] Pade approximant with scaling and squaring.

    Simplifications:

        * we always use a [13/13] approximate
        * no matrix balancing

    References
    ----------
    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)
       https://doi.org/10.1137/04061101X

    