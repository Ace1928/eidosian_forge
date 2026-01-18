from __future__ import print_function
import binascii
from Cryptodome.Util.py3compat import bord, bchr
def _key2bin(s):
    """Convert a key into a string of binary digits"""
    kl = map(lambda x: bord(x), s)
    kl = map(lambda x: binary[x >> 4] + binary[x & 15], kl)
    return ''.join(kl)