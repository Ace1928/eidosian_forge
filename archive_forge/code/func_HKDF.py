import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def HKDF(master, key_len, salt, hashmod, num_keys=1, context=None):
    """Derive one or more keys from a master secret using
    the HMAC-based KDF defined in RFC5869_.

    Args:
     master (byte string):
        The unguessable value used by the KDF to generate the other keys.
        It must be a high-entropy secret, though not necessarily uniform.
        It must not be a password.
     key_len (integer):
        The length in bytes of every derived key.
     salt (byte string):
        A non-secret, reusable value that strengthens the randomness
        extraction step.
        Ideally, it is as long as the digest size of the chosen hash.
        If empty, a string of zeroes in used.
     hashmod (module):
        A cryptographic hash algorithm from :mod:`Cryptodome.Hash`.
        :mod:`Cryptodome.Hash.SHA512` is a good choice.
     num_keys (integer):
        The number of keys to derive. Every key is :data:`key_len` bytes long.
        The maximum cumulative length of all keys is
        255 times the digest size.
     context (byte string):
        Optional identifier describing what the keys are used for.

    Return:
        A byte string or a tuple of byte strings.

    .. _RFC5869: http://tools.ietf.org/html/rfc5869
    """
    output_len = key_len * num_keys
    if output_len > 255 * hashmod.digest_size:
        raise ValueError('Too much secret data to derive')
    if not salt:
        salt = b'\x00' * hashmod.digest_size
    if context is None:
        context = b''
    hmac = HMAC.new(salt, master, digestmod=hashmod)
    prk = hmac.digest()
    t = [b'']
    n = 1
    tlen = 0
    while tlen < output_len:
        hmac = HMAC.new(prk, t[-1] + context + struct.pack('B', n), digestmod=hashmod)
        t.append(hmac.digest())
        tlen += hashmod.digest_size
        n += 1
    derived_output = b''.join(t)
    if num_keys == 1:
        return derived_output[:key_len]
    kol = [derived_output[idx:idx + key_len] for idx in iter_range(0, output_len, key_len)]
    return list(kol[:num_keys])