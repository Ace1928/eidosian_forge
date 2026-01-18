import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def PBKDF2(password, salt, dkLen=16, count=1000, prf=None, hmac_hash_module=None):
    """Derive one or more keys from a password (or passphrase).

    This function performs key derivation according to the PKCS#5 standard (v2.0).

    Args:
     password (string or byte string):
        The secret password to generate the key from.

        Strings will be encoded as ISO 8859-1 (also known as Latin-1),
        which does not allow any characters with codepoints > 255.
     salt (string or byte string):
        A (byte) string to use for better protection from dictionary attacks.
        This value does not need to be kept secret, but it should be randomly
        chosen for each derivation. It is recommended to use at least 16 bytes.

        Strings will be encoded as ISO 8859-1 (also known as Latin-1),
        which does not allow any characters with codepoints > 255.
     dkLen (integer):
        The cumulative length of the keys to produce.

        Due to a flaw in the PBKDF2 design, you should not request more bytes
        than the ``prf`` can output. For instance, ``dkLen`` should not exceed
        20 bytes in combination with ``HMAC-SHA1``.
     count (integer):
        The number of iterations to carry out. The higher the value, the slower
        and the more secure the function becomes.

        You should find the maximum number of iterations that keeps the
        key derivation still acceptable on the slowest hardware you must support.

        Although the default value is 1000, **it is recommended to use at least
        1000000 (1 million) iterations**.
     prf (callable):
        A pseudorandom function. It must be a function that returns a
        pseudorandom byte string from two parameters: a secret and a salt.
        The slower the algorithm, the more secure the derivation function.
        If not specified, **HMAC-SHA1** is used.
     hmac_hash_module (module):
        A module from ``Cryptodome.Hash`` implementing a Merkle-Damgard cryptographic
        hash, which PBKDF2 must use in combination with HMAC.
        This parameter is mutually exclusive with ``prf``.

    Return:
        A byte string of length ``dkLen`` that can be used as key material.
        If you want multiple keys, just break up this string into segments of the desired length.
    """
    password = tobytes(password)
    salt = tobytes(salt)
    if prf and hmac_hash_module:
        raise ValueError("'prf' and 'hmac_hash_module' are mutually exlusive")
    if prf is None and hmac_hash_module is None:
        hmac_hash_module = SHA1
    if prf or not hasattr(hmac_hash_module, '_pbkdf2_hmac_assist'):
        if prf is None:
            prf = lambda p, s: HMAC.new(p, s, hmac_hash_module).digest()

        def link(s):
            s[0], s[1] = (s[1], prf(password, s[1]))
            return s[0]
        key = b''
        i = 1
        while len(key) < dkLen:
            s = [prf(password, salt + struct.pack('>I', i))] * 2
            key += reduce(strxor, (link(s) for j in range(count)))
            i += 1
    else:
        key = b''
        i = 1
        while len(key) < dkLen:
            base = HMAC.new(password, b'', hmac_hash_module)
            first_digest = base.copy().update(salt + struct.pack('>I', i)).digest()
            key += base._pbkdf2_hmac_assist(first_digest, count)
            i += 1
    return key[:dkLen]