from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Util import _cpu_features
def _compute_mac(self):
    """Compute MAC without any FSM checks."""
    if self._tag:
        return self._tag
    self._pad_cache_and_update()
    self._update(long_to_bytes(8 * self._auth_len, 8))
    self._update(long_to_bytes(8 * self._msg_len, 8))
    s_tag = self._signer.digest()
    self._tag = self._tag_cipher.encrypt(s_tag)[:self._mac_len]
    return self._tag