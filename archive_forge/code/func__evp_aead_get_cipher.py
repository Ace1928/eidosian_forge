from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_aead_get_cipher(backend: Backend, cipher: _AEADTypes):
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    assert isinstance(cipher, ChaCha20Poly1305)
    return backend._lib.EVP_aead_chacha20_poly1305()