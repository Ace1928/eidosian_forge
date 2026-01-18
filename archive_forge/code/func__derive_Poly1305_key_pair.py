from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _derive_Poly1305_key_pair(key, nonce):
    """Derive a tuple (r, s, nonce) for a Poly1305 MAC.

    If nonce is ``None``, a new 12-byte nonce is generated.
    """
    if len(key) != 32:
        raise ValueError('Poly1305 with ChaCha20 requires a 32-byte key')
    if nonce is None:
        padded_nonce = nonce = get_random_bytes(12)
    elif len(nonce) == 8:
        padded_nonce = b'\x00\x00\x00\x00' + nonce
    elif len(nonce) == 12:
        padded_nonce = nonce
    else:
        raise ValueError('Poly1305 with ChaCha20 requires an 8- or 12-byte nonce')
    rs = new(key=key, nonce=padded_nonce).encrypt(b'\x00' * 32)
    return (rs[:16], rs[16:], nonce)