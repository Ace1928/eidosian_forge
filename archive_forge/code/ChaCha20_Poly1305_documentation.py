from binascii import unhexlify
from Cryptodome.Cipher import ChaCha20
from Cryptodome.Cipher.ChaCha20 import _HChaCha20
from Cryptodome.Hash import Poly1305, BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import _copy_bytes, bord
from Cryptodome.Util._raw_api import is_buffer
Perform :meth:`decrypt` and :meth:`verify` in one step.

        :param ciphertext: The piece of data to decrypt.
        :type ciphertext: bytes/bytearray/memoryview
        :param bytes received_mac_tag:
            This is the 16-byte *binary* MAC, as received from the sender.
        :return: the decrypted data (as ``bytes``)
        :raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        