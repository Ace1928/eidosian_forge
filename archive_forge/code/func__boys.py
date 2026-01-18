import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _boys(n, t):
    """Evaluate the Boys function.

    The :math:`n`-th order `Boys function <https://arxiv.org/abs/2107.01488>`_ is defined as

    .. math::

        F_n(t) = \\int_{0}^{1}x^{2n} e^{-tx^2}dx.

    The Boys function is related to the lower incomplete Gamma
    `function <https://en.wikipedia.org/wiki/Incomplete_gamma_function>`_, :math:`\\gamma`, as

    .. math::

        F_n(t) = \\frac{1}{2t^{n + 0.5}} \\gamma(n + 0.5, t),

    where

    .. math::

        \\gamma(m, t) = \\int_{0}^{t} x^{m-1} e^{-x} dx.

    Args:
        n (float): order of the Boys function
        t (array[float]): exponent of the Boys function

    Returns:
        (array[float]): value of the Boys function
    """
    return qml.math.where(t == 0.0, 1 / (2 * n + 1), qml.math.gammainc(n + 0.5, t + (t == 0.0)) * qml.math.gamma(n + 0.5) / (2 * (t + (t == 0.0)) ** (n + 0.5)))