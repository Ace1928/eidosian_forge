from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, tobytes, is_bytes
from Cryptodome.Random import get_random_bytes
from . import cSHAKE128, SHA3_256
from .cSHAKE128 import _bytepad, _encode_str, _right_encode
Return a new instance of a KMAC hash object.
        See :func:`new`.
        