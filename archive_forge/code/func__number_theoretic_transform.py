from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int
def _number_theoretic_transform(seq, prime, inverse=False):
    """Utility function for the Number Theoretic Transform"""
    if not iterable(seq):
        raise TypeError('Expected a sequence of integer coefficients for Number Theoretic Transform')
    p = as_int(prime)
    if not isprime(p):
        raise ValueError('Expected prime modulus for Number Theoretic Transform')
    a = [as_int(x) % p for x in seq]
    n = len(a)
    if n < 1:
        return a
    b = n.bit_length() - 1
    if n & n - 1:
        b += 1
        n = 2 ** b
    if (p - 1) % n:
        raise ValueError('Expected prime modulus of the form (m*2**k + 1)')
    a += [0] * (n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = (a[j], a[i])
    pr = primitive_root(p)
    rt = pow(pr, (p - 1) // n, p)
    if inverse:
        rt = pow(rt, p - 2, p)
    w = [1] * (n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1] * rt % p
    h = 2
    while h <= n:
        hf, ut = (h // 2, n // h)
        for i in range(0, n, h):
            for j in range(hf):
                u, v = (a[i + j], a[i + j + hf] * w[ut * j])
                a[i + j], a[i + j + hf] = ((u + v) % p, (u - v) % p)
        h *= 2
    if inverse:
        rv = pow(n, p - 2, p)
        a = [x * rv % p for x in a]
    return a