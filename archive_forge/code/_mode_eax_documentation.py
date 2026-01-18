import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import byte_string, bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import CMAC, BLAKE2s
from Cryptodome.Random import get_random_bytes
Perform decrypt() and verify() in one step.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
          received_mac_tag : bytes/bytearray/memoryview
            This is the *binary* MAC, as received from the sender.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return: the plaintext as ``bytes`` or ``None`` when the ``output``
            parameter specified a location for the result.
        :Raises MacMismatchError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        