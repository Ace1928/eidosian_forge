from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_set_length(backend: Backend, ctx, data_len: int) -> None:
    intptr = backend._ffi.new('int *')
    res = backend._lib.EVP_CipherUpdate(ctx, backend._ffi.NULL, intptr, backend._ffi.NULL, data_len)
    backend.openssl_assert(res != 0)