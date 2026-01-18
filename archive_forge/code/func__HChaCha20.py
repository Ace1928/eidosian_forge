from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _HChaCha20(key, nonce):
    assert len(key) == 32
    assert len(nonce) == 16
    subkey = bytearray(32)
    result = _raw_chacha20_lib.hchacha20(c_uint8_ptr(key), c_uint8_ptr(nonce), c_uint8_ptr(subkey))
    if result:
        raise ValueError('Error %d when deriving subkey with HChaCha20' % result)
    return subkey