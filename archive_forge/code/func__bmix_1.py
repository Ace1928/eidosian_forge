import operator
import struct
from passlib.utils.compat import izip
from passlib.crypto.digest import pbkdf2_hmac
from passlib.crypto.scrypt._salsa import salsa20
def _bmix_1(self, source, target):
    """special bmix() method optimized for ``r=1`` case"""
    B = source[16:]
    target[:16] = tmp = salsa20((a ^ b for a, b in izip(B, iter(source))))
    target[16:] = salsa20((a ^ b for a, b in izip(tmp, B)))