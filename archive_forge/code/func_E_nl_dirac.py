from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm
def E_nl_dirac(n, l, spin_up=True, Z=1, c=Float('137.035999037')):
    """
    Returns the relativistic energy of the state (n, l, spin) in Hartree atomic
    units.

    The energy is calculated from the Dirac equation. The rest mass energy is
    *not* included.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    spin_up :
        True if the electron spin is up (default), otherwise down
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)
    c :
        Speed of light in atomic units. Default value is 137.035999037,
        taken from https://arxiv.org/abs/1012.3627

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl_dirac
    >>> E_nl_dirac(1, 0)
    -0.500006656595360

    >>> E_nl_dirac(2, 0)
    -0.125002080189006
    >>> E_nl_dirac(2, 1)
    -0.125000416028342
    >>> E_nl_dirac(2, 1, False)
    -0.125002080189006

    >>> E_nl_dirac(3, 0)
    -0.0555562951740285
    >>> E_nl_dirac(3, 1)
    -0.0555558020932949
    >>> E_nl_dirac(3, 1, False)
    -0.0555562951740285
    >>> E_nl_dirac(3, 2)
    -0.0555556377366884
    >>> E_nl_dirac(3, 2, False)
    -0.0555558020932949

    """
    n, l, Z, c = map(S, [n, l, Z, c])
    if not l >= 0:
        raise ValueError("'l' must be positive or zero")
    if not n > l:
        raise ValueError("'n' must be greater than 'l'")
    if l == 0 and spin_up is False:
        raise ValueError('Spin must be up for l==0.')
    if spin_up:
        skappa = -l - 1
    else:
        skappa = -l
    beta = sqrt(skappa ** 2 - Z ** 2 / c ** 2)
    return c ** 2 / sqrt(1 + Z ** 2 / (n + skappa + beta) ** 2 / c ** 2) - c ** 2