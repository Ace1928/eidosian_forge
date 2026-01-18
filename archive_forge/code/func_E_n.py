from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar
def E_n(n, omega):
    """
    Returns the Energy of the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    omega :
        The harmonic oscillator angular frequency.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_n = hbar * omega*(n + 1/2)

    Examples
    ========

    >>> from sympy.physics.qho_1d import E_n
    >>> from sympy.abc import x, omega
    >>> E_n(x, omega)
    hbar*omega*(x + 1/2)
    """
    return hbar * omega * (n + S.Half)