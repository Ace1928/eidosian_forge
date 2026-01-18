from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def _raise_mod_power(x, s, p, f):
    """
    Used in gf_csolve to generate solutions of f(x) cong 0 mod(p**(s + 1))
    from the solutions of f(x) cong 0 mod(p**s).

    Examples
    ========

    >>> from sympy.polys.galoistools import _raise_mod_power
    >>> from sympy.polys.galoistools import csolve_prime

    These is the solutions of f(x) = x**2 + x + 7 cong 0 mod(3)

    >>> f = [1, 1, 7]
    >>> csolve_prime(f, 3)
    [1]
    >>> [ i for i in range(3) if not (i**2 + i + 7) % 3]
    [1]

    The solutions of f(x) cong 0 mod(9) are constructed from the
    values returned from _raise_mod_power:

    >>> x, s, p = 1, 1, 3
    >>> V = _raise_mod_power(x, s, p, f)
    >>> [x + v * p**s for v in V]
    [1, 4, 7]

    And these are confirmed with the following:

    >>> [ i for i in range(3**2) if not (i**2 + i + 7) % 3**2]
    [1, 4, 7]

    """
    from sympy.polys.domains import ZZ
    f_f = gf_diff(f, p, ZZ)
    alpha = gf_value(f_f, x)
    beta = -gf_value(f, x) // p ** s
    return linear_congruence(alpha, beta, p)