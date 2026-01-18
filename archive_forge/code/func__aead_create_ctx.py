from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _aead_create_ctx(backend: Backend, cipher: _AEADTypes, key: bytes):
    if _is_evp_aead_supported_cipher(backend, cipher):
        return _evp_aead_create_ctx(backend, cipher, key)
    else:
        return _evp_cipher_create_ctx(backend, cipher, key)