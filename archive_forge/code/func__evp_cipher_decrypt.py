from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_decrypt(backend: Backend, cipher: _AEADTypes, nonce: bytes, data: bytes, associated_data: typing.List[bytes], tag_length: int, ctx: typing.Any=None) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESCCM, AESSIV
    if len(data) < tag_length:
        raise InvalidTag
    if isinstance(cipher, AESSIV):
        tag = data[:tag_length]
        data = data[tag_length:]
    else:
        tag = data[-tag_length:]
        data = data[:-tag_length]
    if ctx is None:
        cipher_name = _evp_cipher_cipher_name(cipher)
        ctx = _evp_cipher_aead_setup(backend, cipher_name, cipher._key, nonce, tag, tag_length, _DECRYPT)
    else:
        _evp_cipher_set_nonce_operation(backend, ctx, nonce, _DECRYPT)
        _evp_cipher_set_tag(backend, ctx, tag)
    if isinstance(cipher, AESCCM):
        _evp_cipher_set_length(backend, ctx, len(data))
    for ad in associated_data:
        _evp_cipher_process_aad(backend, ctx, ad)
    if isinstance(cipher, AESCCM):
        outlen = backend._ffi.new('int *')
        buf = backend._ffi.new('unsigned char[]', len(data))
        d_ptr = backend._ffi.from_buffer(data)
        res = backend._lib.EVP_CipherUpdate(ctx, buf, outlen, d_ptr, len(data))
        if res != 1:
            backend._consume_errors()
            raise InvalidTag
        processed_data = backend._ffi.buffer(buf, outlen[0])[:]
    else:
        processed_data = _evp_cipher_process_data(backend, ctx, data)
        outlen = backend._ffi.new('int *')
        buf = backend._ffi.new('unsigned char[]', 16)
        res = backend._lib.EVP_CipherFinal_ex(ctx, buf, outlen)
        processed_data += backend._ffi.buffer(buf, outlen[0])[:]
        if res == 0:
            backend._consume_errors()
            raise InvalidTag
    return processed_data