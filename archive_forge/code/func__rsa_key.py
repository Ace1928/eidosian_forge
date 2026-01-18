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
def _rsa_key(*args, public=True, private=True, totient='Euler', index=None, multipower=None):
    """A private subroutine to generate RSA key

    Parameters
    ==========

    public, private : bool, optional
        Flag to generate either a public key, a private key.

    totient : 'Euler' or 'Carmichael'
        Different notation used for totient.

    multipower : bool, optional
        Flag to bypass warning for multipower RSA.
    """
    if len(args) < 2:
        return False
    if totient not in ('Euler', 'Carmichael'):
        raise ValueError("The argument totient={} should either be 'Euler', 'Carmichalel'.".format(totient))
    if totient == 'Euler':
        _totient = _euler
    else:
        _totient = _carmichael
    if index is not None:
        index = as_int(index)
        if totient != 'Carmichael':
            raise ValueError("Setting the 'index' keyword argument requires totientnotation to be specified as 'Carmichael'.")
    primes, e = (args[:-1], args[-1])
    if not all((isprime(p) for p in primes)):
        new_primes = []
        for i in primes:
            new_primes.extend(factorint(i, multiple=True))
        primes = new_primes
    n = reduce(lambda i, j: i * j, primes)
    tally = multiset(primes)
    if all((v == 1 for v in tally.values())):
        multiple = list(tally.keys())
        phi = _totient._from_distinct_primes(*multiple)
    else:
        if not multipower:
            NonInvertibleCipherWarning('Non-distinctive primes found in the factors {}. The cipher may not be decryptable for some numbers in the complete residue system Z[{}], but the cipher can still be valid if you restrict the domain to be the reduced residue system Z*[{}]. You can pass the flag multipower=True if you want to suppress this warning.'.format(primes, n, n)).warn(stacklevel=4)
        phi = _totient._from_factors(tally)
    if igcd(e, phi) == 1:
        if public and (not private):
            if isinstance(index, int):
                e = e % phi
                e += index * phi
            return (n, e)
        if private and (not public):
            d = mod_inverse(e, phi)
            if isinstance(index, int):
                d += index * phi
            return (n, d)
    return False