import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _initial_nodes_a(n, k):
    """Tricomi initial guesses

    Computes an initial approximation to the square of the `k`-th
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.1 in the
    original paper. The guesses are accurate except in the region
    near :math:`\\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate roots

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    tauk = _compute_tauk(n, k)
    sigk = cos(0.5 * tauk) ** 2
    a = n % 2 - 0.5
    nu = 4.0 * floor(n / 2.0) + 2.0 * a + 2.0
    xksq = nu * sigk - 1.0 / (3.0 * nu) * (5.0 / (4.0 * (1.0 - sigk) ** 2) - 1.0 / (1.0 - sigk) - 0.25)
    return xksq