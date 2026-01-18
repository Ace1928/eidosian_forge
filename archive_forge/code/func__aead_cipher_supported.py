from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _aead_cipher_supported(backend: Backend, cipher: _AEADTypes) -> bool:
    if _is_evp_aead_supported_cipher(backend, cipher):
        return True
    else:
        cipher_name = _evp_cipher_cipher_name(cipher)
        if backend._fips_enabled and cipher_name not in backend._fips_aead:
            return False
        if cipher_name.endswith(b'-siv'):
            return backend._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER == 1
        else:
            return backend._lib.EVP_get_cipherbyname(cipher_name) != backend._ffi.NULL