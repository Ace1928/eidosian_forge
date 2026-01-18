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
def AZ(s=None):
    """Return the letters of ``s`` in uppercase. In case more than
    one string is passed, each of them will be processed and a list
    of upper case strings will be returned.

    Examples
    ========

    >>> from sympy.crypto.crypto import AZ
    >>> AZ('Hello, world!')
    'HELLOWORLD'
    >>> AZ('Hello, world!'.split())
    ['HELLO', 'WORLD']

    See Also
    ========

    check_and_join

    """
    if not s:
        return uppercase
    t = isinstance(s, str)
    if t:
        s = [s]
    rv = [check_and_join(i.upper().split(), uppercase, filter=True) for i in s]
    if t:
        return rv[0]
    return rv