from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Util import _cpu_features
class _GHASH(object):
    """GHASH function defined in NIST SP 800-38D, Algorithm 2.

    If X_1, X_2, .. X_m are the blocks of input data, the function
    computes:

       X_1*H^{m} + X_2*H^{m-1} + ... + X_m*H

    in the Galois field GF(2^256) using the reducing polynomial
    (x^128 + x^7 + x^2 + x + 1).
    """

    def __init__(self, subkey, ghash_c):
        assert len(subkey) == 16
        self.ghash_c = ghash_c
        self._exp_key = VoidPointer()
        result = ghash_c.ghash_expand(c_uint8_ptr(subkey), self._exp_key.address_of())
        if result:
            raise ValueError('Error %d while expanding the GHASH key' % result)
        self._exp_key = SmartPointer(self._exp_key.get(), ghash_c.ghash_destroy)
        self._last_y = create_string_buffer(16)

    def update(self, block_data):
        assert len(block_data) % 16 == 0
        result = self.ghash_c.ghash(self._last_y, c_uint8_ptr(block_data), c_size_t(len(block_data)), self._last_y, self._exp_key.get())
        if result:
            raise ValueError('Error %d while updating GHASH' % result)
        return self

    def digest(self):
        return get_raw_buffer(self._last_y)