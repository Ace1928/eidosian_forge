from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def _legendre(a, p):
    """
    Returns the legendre symbol of a and p
    assuming that p is a prime.

    i.e. 1 if a is a quadratic residue mod p
        -1 if a is not a quadratic residue mod p
         0 if a is divisible by p

    Parameters
    ==========

    a : int
        The number to test.

    p : prime
        The prime to test ``a`` against.

    Returns
    =======

    int
        Legendre symbol (a / p).

    """
    sig = pow(a, (p - 1) // 2, p)
    if sig == 1:
        return 1
    elif sig == 0:
        return 0
    else:
        return -1