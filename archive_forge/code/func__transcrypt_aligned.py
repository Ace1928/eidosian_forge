import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes, bchr
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
def _transcrypt_aligned(self, in_data, in_data_len, trans_func, trans_desc):
    out_data = create_string_buffer(in_data_len)
    result = trans_func(self._state.get(), in_data, out_data, c_size_t(in_data_len))
    if result:
        raise ValueError('Error %d while %sing in OCB mode' % (result, trans_desc))
    return get_raw_buffer(out_data)