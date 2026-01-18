from binascii import unhexlify
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import bord, tobytes, _copy_bytes
from Cryptodome.Random import get_random_bytes
Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
          hex_mac_tag (string): the expected MAC of the message, as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        