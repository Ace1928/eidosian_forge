import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _bcrypt_decode(data):
    s = './ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    bits = []
    for c in tostr(data):
        idx = s.find(c)
        bits6 = bin(idx)[2:].zfill(6)
        bits.append(bits6)
    bits = ''.join(bits)
    modulo4 = len(data) % 4
    if modulo4 == 1:
        raise ValueError('Incorrect length')
    elif modulo4 == 2:
        bits = bits[:-4]
    elif modulo4 == 3:
        bits = bits[:-2]
    bits8 = [bits[idx:idx + 8] for idx in range(0, len(bits), 8)]
    result = []
    for g in bits8:
        result.append(bchr(int(g, 2)))
    result = b''.join(result)
    return result