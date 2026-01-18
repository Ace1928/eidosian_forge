from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Protocol.KDF import _S2V
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
def _create_siv_cipher(factory, **kwargs):
    """Create a new block cipher, configured in
    Synthetic Initializaton Vector (SIV) mode.

    :Parameters:

      factory : object
        A symmetric cipher module from `Cryptodome.Cipher`
        (like `Cryptodome.Cipher.AES`).

    :Keywords:

      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.
        It must be 32, 48 or 64 bytes long.
        If AES is the chosen cipher, the variants *AES-128*,
        *AES-192* and or *AES-256* will be used internally.

      nonce : bytes/bytearray/memoryview
        For deterministic encryption, it is not present.

        Otherwise, it is a value that must never be reused
        for encrypting message under this key.

        There are no restrictions on its length,
        but it is recommended to use at least 16 bytes.
    """
    try:
        key = kwargs.pop('key')
    except KeyError as e:
        raise TypeError('Missing parameter: ' + str(e))
    nonce = kwargs.pop('nonce', None)
    return SivMode(factory, key, nonce, kwargs)