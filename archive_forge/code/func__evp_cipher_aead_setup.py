from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_aead_setup(backend: Backend, cipher_name: bytes, key: bytes, nonce: bytes, tag: typing.Optional[bytes], tag_len: int, operation: int):
    evp_cipher = _evp_cipher(cipher_name, backend)
    ctx = backend._lib.EVP_CIPHER_CTX_new()
    ctx = backend._ffi.gc(ctx, backend._lib.EVP_CIPHER_CTX_free)
    res = backend._lib.EVP_CipherInit_ex(ctx, evp_cipher, backend._ffi.NULL, backend._ffi.NULL, backend._ffi.NULL, int(operation == _ENCRYPT))
    backend.openssl_assert(res != 0)
    res = backend._lib.EVP_CIPHER_CTX_ctrl(ctx, backend._lib.EVP_CTRL_AEAD_SET_IVLEN, len(nonce), backend._ffi.NULL)
    backend.openssl_assert(res != 0)
    if operation == _DECRYPT:
        assert tag is not None
        _evp_cipher_set_tag(backend, ctx, tag)
    elif cipher_name.endswith(b'-ccm'):
        res = backend._lib.EVP_CIPHER_CTX_ctrl(ctx, backend._lib.EVP_CTRL_AEAD_SET_TAG, tag_len, backend._ffi.NULL)
        backend.openssl_assert(res != 0)
    nonce_ptr = backend._ffi.from_buffer(nonce)
    key_ptr = backend._ffi.from_buffer(key)
    res = backend._lib.EVP_CipherInit_ex(ctx, backend._ffi.NULL, backend._ffi.NULL, key_ptr, nonce_ptr, int(operation == _ENCRYPT))
    backend.openssl_assert(res != 0)
    return ctx