from Cryptodome.Util.py3compat import is_bytes
from .KMAC128 import KMAC_Hash
from . import cSHAKE256
Create a new KMAC256 object.

    Args:
        key (bytes/bytearray/memoryview):
            The key to use to compute the MAC.
            It must be at least 256 bits long (32 bytes).
        data (bytes/bytearray/memoryview):
            Optional. The very first chunk of the message to authenticate.
            It is equivalent to an early call to :meth:`KMAC_Hash.update`.
        mac_len (integer):
            Optional. The size of the authentication tag, in bytes.
            Default is 64. Minimum is 8.
        custom (bytes/bytearray/memoryview):
            Optional. A customization byte string (``S`` in SP 800-185).

    Returns:
        A :class:`KMAC_Hash` hash object
    