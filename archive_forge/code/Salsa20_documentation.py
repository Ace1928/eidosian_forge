from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Random import get_random_bytes
Decrypt a piece of data.
        
        Args:
          ciphertext(bytes/bytearray/memoryview): The data to decrypt, of any size.
        Keyword Args:
          output(bytes/bytearray/memoryview): The location where the plaintext
            is written to. If ``None``, the plaintext is returned.
        Returns:
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        