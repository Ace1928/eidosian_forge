import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes, bchr
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
def _create_ocb_cipher(factory, **kwargs):
    """Create a new block cipher, configured in OCB mode.

    :Parameters:
      factory : module
        A symmetric cipher module from `Cryptodome.Cipher`
        (like `Cryptodome.Cipher.AES`).

    :Keywords:
      nonce : bytes/bytearray/memoryview
        A  value that must never be reused for any other encryption.
        Its length can vary from 1 to 15 bytes.
        If not specified, a random 15 bytes long nonce is generated.

      mac_len : integer
        Length of the MAC, in bytes.
        It must be in the range ``[8..16]``.
        The default is 16 (128 bits).

    Any other keyword will be passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present).
    """
    try:
        nonce = kwargs.pop('nonce', None)
        if nonce is None:
            nonce = get_random_bytes(15)
        mac_len = kwargs.pop('mac_len', 16)
    except KeyError as e:
        raise TypeError('Keyword missing: ' + str(e))
    return OcbMode(factory, nonce, mac_len, kwargs)