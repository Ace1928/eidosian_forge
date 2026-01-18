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
def _help(m, prime_modulo_method, diff_method, expr_val):
    """
    Helper function for _nthroot_mod_composite and polynomial_congruence.

    Parameters
    ==========

    m : positive integer
    prime_modulo_method : function to calculate the root of the congruence
    equation for the prime divisors of m
    diff_method : function to calculate derivative of expression at any
    given point
    expr_val : function to calculate value of the expression at any
    given point
    """
    from sympy.ntheory.modular import crt
    f = factorint(m)
    dd = {}
    for p, e in f.items():
        tot_roots = set()
        if e == 1:
            tot_roots.update(prime_modulo_method(p))
        else:
            for root in prime_modulo_method(p):
                diff = diff_method(root, p)
                if diff != 0:
                    ppow = p
                    m_inv = mod_inverse(diff, p)
                    for j in range(1, e):
                        ppow *= p
                        root = (root - expr_val(root, ppow) * m_inv) % ppow
                    tot_roots.add(root)
                else:
                    new_base = p
                    roots_in_base = {root}
                    while new_base < pow(p, e):
                        new_base *= p
                        new_roots = set()
                        for k in roots_in_base:
                            if expr_val(k, new_base) != 0:
                                continue
                            while k not in new_roots:
                                new_roots.add(k)
                                k = (k + new_base // p) % new_base
                        roots_in_base = new_roots
                    tot_roots = tot_roots | roots_in_base
        if tot_roots == set():
            return []
        dd[pow(p, e)] = tot_roots
    a = []
    m = []
    for x, y in dd.items():
        m.append(x)
        a.append(list(y))
    return sorted({crt(m, list(i))[0] for i in product(*a)})