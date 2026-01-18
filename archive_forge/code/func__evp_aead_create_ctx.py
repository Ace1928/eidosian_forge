from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_aead_create_ctx(backend: Backend, cipher: _AEADTypes, key: bytes, tag_len: typing.Optional[int]=None):
    aead_cipher = _evp_aead_get_cipher(backend, cipher)
    assert aead_cipher is not None
    key_ptr = backend._ffi.from_buffer(key)
    tag_len = backend._lib.EVP_AEAD_DEFAULT_TAG_LENGTH if tag_len is None else tag_len
    ctx = backend._lib.Cryptography_EVP_AEAD_CTX_new(aead_cipher, key_ptr, len(key), tag_len)
    backend.openssl_assert(ctx != backend._ffi.NULL)
    ctx = backend._ffi.gc(ctx, backend._lib.EVP_AEAD_CTX_free)
    return ctx