from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def _nthroot_mod1(s, q, p, all_roots):
    """
    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``

    References
    ==========

    .. [1] A. M. Johnston "A Generalized qth Root Algorithm"

    """
    g = primitive_root(p)
    if not isprime(q):
        r = _nthroot_mod2(s, q, p)
    else:
        f = p - 1
        assert (p - 1) % q == 0
        k = 0
        while f % q == 0:
            k += 1
            f = f // q
        f1 = igcdex(-f, q)[0] % q
        z = f * f1
        x = (1 + z) // q
        r1 = pow(s, x, p)
        s1 = pow(s, f, p)
        h = pow(g, f * q, p)
        t = discrete_log(p, s1, h)
        g2 = pow(g, z * t, p)
        g3 = igcdex(g2, p)[0]
        r = r1 * g3 % p
    res = [r]
    h = pow(g, (p - 1) // q, p)
    hx = r
    for i in range(q - 1):
        hx = hx * h % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)