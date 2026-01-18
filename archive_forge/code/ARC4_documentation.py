from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
Decrypt a piece of data.

        :param ciphertext: The data to decrypt, of any size.
        :type ciphertext: bytes, bytearray, memoryview
        :returns: the decrypted byte string, of equal length as the
          ciphertext.
        