from binascii import unhexlify
from Cryptodome.Cipher import ChaCha20
from Cryptodome.Cipher.ChaCha20 import _HChaCha20
from Cryptodome.Hash import Poly1305, BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import _copy_bytes, bord
from Cryptodome.Util._raw_api import is_buffer
def _pad_aad(self):
    assert self._status == _CipherStatus.PROCESSING_AUTH_DATA
    if self._len_aad & 15:
        self._authenticator.update(b'\x00' * (16 - (self._len_aad & 15)))
    self._status = _CipherStatus.PROCESSING_CIPHERTEXT