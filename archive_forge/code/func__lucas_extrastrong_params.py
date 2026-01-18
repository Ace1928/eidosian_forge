from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def _lucas_extrastrong_params(n):
    """Calculates the "extra strong" parameters (D, P, Q) for n.

    References
    ==========
    .. [1] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [1] https://en.wikipedia.org/wiki/Lucas_pseudoprime
    """
    from sympy.ntheory.residue_ntheory import jacobi_symbol
    P, Q, D = (3, 1, 5)
    while True:
        g = igcd(D, n)
        if g > 1 and g != n:
            return (0, 0, 0)
        if jacobi_symbol(D, n) == -1:
            break
        P += 1
        D = P * P - 4
    return _int_tuple(D, P, Q)