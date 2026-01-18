import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes, bchr
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
def _compute_mac_tag(self):
    if self._mac_tag is not None:
        return
    if self._cache_A:
        self._update(self._cache_A, len(self._cache_A))
        self._cache_A = b''
    mac_tag = create_string_buffer(16)
    result = _raw_ocb_lib.OCB_digest(self._state.get(), mac_tag, c_size_t(len(mac_tag)))
    if result:
        raise ValueError('Error %d while computing digest in OCB mode' % result)
    self._mac_tag = get_raw_buffer(mac_tag)[:self._mac_len]