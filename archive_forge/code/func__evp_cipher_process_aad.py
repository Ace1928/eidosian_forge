from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_process_aad(backend: Backend, ctx, associated_data: bytes) -> None:
    outlen = backend._ffi.new('int *')
    a_data_ptr = backend._ffi.from_buffer(associated_data)
    res = backend._lib.EVP_CipherUpdate(ctx, backend._ffi.NULL, outlen, a_data_ptr, len(associated_data))
    backend.openssl_assert(res != 0)