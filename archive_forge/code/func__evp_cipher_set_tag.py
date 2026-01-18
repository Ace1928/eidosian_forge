from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_set_tag(backend, ctx, tag: bytes) -> None:
    tag_ptr = backend._ffi.from_buffer(tag)
    res = backend._lib.EVP_CIPHER_CTX_ctrl(ctx, backend._lib.EVP_CTRL_AEAD_SET_TAG, len(tag), tag_ptr)
    backend.openssl_assert(res != 0)