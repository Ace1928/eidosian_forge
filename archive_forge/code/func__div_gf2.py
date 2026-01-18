from Cryptodome.Util.py3compat import is_native_int
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Random import get_random_bytes as rng
def _div_gf2(a, b):
    """
    Compute division of polynomials over GF(2).
    Given a and b, it finds two polynomials q and r such that:

    a = b*q + r with deg(r)<deg(b)
    """
    if a < b:
        return (0, a)
    deg = number.size
    q = 0
    r = a
    d = deg(b)
    while deg(r) >= d:
        s = 1 << deg(r) - d
        q ^= s
        r ^= _mult_gf2(b, s)
    return (q, r)