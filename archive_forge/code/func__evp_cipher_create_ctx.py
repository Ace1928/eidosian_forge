from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_create_ctx(backend: Backend, cipher: _AEADTypes, key: bytes):
    ctx = backend._lib.EVP_CIPHER_CTX_new()
    backend.openssl_assert(ctx != backend._ffi.NULL)
    ctx = backend._ffi.gc(ctx, backend._lib.EVP_CIPHER_CTX_free)
    cipher_name = _evp_cipher_cipher_name(cipher)
    evp_cipher = _evp_cipher(cipher_name, backend)
    key_ptr = backend._ffi.from_buffer(key)
    res = backend._lib.EVP_CipherInit_ex(ctx, evp_cipher, backend._ffi.NULL, key_ptr, backend._ffi.NULL, 0)
    backend.openssl_assert(res != 0)
    return ctx