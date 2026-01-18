from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _is_evp_aead_supported_cipher(backend: Backend, cipher: _AEADTypes) -> bool:
    """
    Checks whether the given cipher is supported through
    EVP_AEAD rather than the normal OpenSSL EVP_CIPHER API.
    """
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    return backend._lib.Cryptography_HAS_EVP_AEAD and isinstance(cipher, ChaCha20Poly1305)