import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers, shift, scale):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) float ndarray

    """
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale
    vec = cp.empty((q, p + r), dtype=float)
    delta = xeps[:, None, :] - yeps[None, :, :]
    vec[:, :p] = kernel_func(cp.linalg.norm(delta, axis=-1))
    pwr = xhat[:, None, :] ** powers[None, :, :]
    vec[:, p:] = cp.prod(pwr, axis=-1)
    return vec