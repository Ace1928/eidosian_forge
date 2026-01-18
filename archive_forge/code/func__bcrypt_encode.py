import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _bcrypt_encode(data):
    s = './ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    bits = []
    for c in data:
        bits_c = bin(bord(c))[2:].zfill(8)
        bits.append(bstr(bits_c))
    bits = b''.join(bits)
    bits6 = [bits[idx:idx + 6] for idx in range(0, len(bits), 6)]
    result = []
    for g in bits6[:-1]:
        idx = int(g, 2)
        result.append(s[idx])
    g = bits6[-1]
    idx = int(g, 2) << 6 - len(g)
    result.append(s[idx])
    result = ''.join(result)
    return tobytes(result)