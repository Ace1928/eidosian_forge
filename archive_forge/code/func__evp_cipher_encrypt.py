from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_encrypt(backend: Backend, cipher: _AEADTypes, nonce: bytes, data: bytes, associated_data: typing.List[bytes], tag_length: int, ctx: typing.Any=None) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESCCM, AESSIV
    if ctx is None:
        cipher_name = _evp_cipher_cipher_name(cipher)
        ctx = _evp_cipher_aead_setup(backend, cipher_name, cipher._key, nonce, None, tag_length, _ENCRYPT)
    else:
        _evp_cipher_set_nonce_operation(backend, ctx, nonce, _ENCRYPT)
    if isinstance(cipher, AESCCM):
        _evp_cipher_set_length(backend, ctx, len(data))
    for ad in associated_data:
        _evp_cipher_process_aad(backend, ctx, ad)
    processed_data = _evp_cipher_process_data(backend, ctx, data)
    outlen = backend._ffi.new('int *')
    buf = backend._ffi.new('unsigned char[]', 16)
    res = backend._lib.EVP_CipherFinal_ex(ctx, buf, outlen)
    backend.openssl_assert(res != 0)
    processed_data += backend._ffi.buffer(buf, outlen[0])[:]
    tag_buf = backend._ffi.new('unsigned char[]', tag_length)
    res = backend._lib.EVP_CIPHER_CTX_ctrl(ctx, backend._lib.EVP_CTRL_AEAD_GET_TAG, tag_length, tag_buf)
    backend.openssl_assert(res != 0)
    tag = backend._ffi.buffer(tag_buf)[:]
    if isinstance(cipher, AESSIV):
        backend.openssl_assert(len(tag) == 16)
        return tag + processed_data
    else:
        return processed_data + tag