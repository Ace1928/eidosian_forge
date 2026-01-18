from Cryptodome.Util.py3compat import is_native_int
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Random import get_random_bytes as rng
def _mult_gf2(f1, f2):
    """Multiply two polynomials in GF(2)"""
    if f2 > f1:
        f1, f2 = (f2, f1)
    z = 0
    while f2:
        if f2 & 1:
            z ^= f1
        f1 <<= 1
        f2 >>= 1
    return z