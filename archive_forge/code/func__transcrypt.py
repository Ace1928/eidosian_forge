import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes, bchr
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
def _transcrypt(self, in_data, trans_func, trans_desc):
    if in_data is None:
        out_data = self._transcrypt_aligned(self._cache_P, len(self._cache_P), trans_func, trans_desc)
        self._cache_P = b''
        return out_data
    prefix = b''
    if len(self._cache_P) > 0:
        filler = min(16 - len(self._cache_P), len(in_data))
        self._cache_P += _copy_bytes(None, filler, in_data)
        in_data = in_data[filler:]
        if len(self._cache_P) < 16:
            return b''
        prefix = self._transcrypt_aligned(self._cache_P, len(self._cache_P), trans_func, trans_desc)
        self._cache_P = b''
    trans_len = len(in_data) // 16 * 16
    result = self._transcrypt_aligned(c_uint8_ptr(in_data), trans_len, trans_func, trans_desc)
    if prefix:
        result = prefix + result
    self._cache_P = _copy_bytes(trans_len, None, in_data)
    return result