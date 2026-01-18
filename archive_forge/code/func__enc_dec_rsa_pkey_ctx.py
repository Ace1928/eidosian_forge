from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _enc_dec_rsa_pkey_ctx(backend: Backend, key: typing.Union[_RSAPrivateKey, _RSAPublicKey], data: bytes, padding_enum: int, padding: AsymmetricPadding) -> bytes:
    init: typing.Callable[[typing.Any], int]
    crypt: typing.Callable[[typing.Any, typing.Any, int, bytes, int], int]
    if isinstance(key, _RSAPublicKey):
        init = backend._lib.EVP_PKEY_encrypt_init
        crypt = backend._lib.EVP_PKEY_encrypt
    else:
        init = backend._lib.EVP_PKEY_decrypt_init
        crypt = backend._lib.EVP_PKEY_decrypt
    pkey_ctx = backend._lib.EVP_PKEY_CTX_new(key._evp_pkey, backend._ffi.NULL)
    backend.openssl_assert(pkey_ctx != backend._ffi.NULL)
    pkey_ctx = backend._ffi.gc(pkey_ctx, backend._lib.EVP_PKEY_CTX_free)
    res = init(pkey_ctx)
    backend.openssl_assert(res == 1)
    res = backend._lib.EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, padding_enum)
    backend.openssl_assert(res > 0)
    buf_size = backend._lib.EVP_PKEY_size(key._evp_pkey)
    backend.openssl_assert(buf_size > 0)
    if isinstance(padding, OAEP):
        mgf1_md = backend._evp_md_non_null_from_algorithm(padding._mgf._algorithm)
        res = backend._lib.EVP_PKEY_CTX_set_rsa_mgf1_md(pkey_ctx, mgf1_md)
        backend.openssl_assert(res > 0)
        oaep_md = backend._evp_md_non_null_from_algorithm(padding._algorithm)
        res = backend._lib.EVP_PKEY_CTX_set_rsa_oaep_md(pkey_ctx, oaep_md)
        backend.openssl_assert(res > 0)
    if isinstance(padding, OAEP) and padding._label is not None and (len(padding._label) > 0):
        labelptr = backend._lib.OPENSSL_malloc(len(padding._label))
        backend.openssl_assert(labelptr != backend._ffi.NULL)
        backend._ffi.memmove(labelptr, padding._label, len(padding._label))
        res = backend._lib.EVP_PKEY_CTX_set0_rsa_oaep_label(pkey_ctx, labelptr, len(padding._label))
        backend.openssl_assert(res == 1)
    outlen = backend._ffi.new('size_t *', buf_size)
    buf = backend._ffi.new('unsigned char[]', buf_size)
    res = crypt(pkey_ctx, buf, outlen, data, len(data))
    resbuf = backend._ffi.buffer(buf)[:outlen[0]]
    backend._lib.ERR_clear_error()
    if res <= 0:
        raise ValueError('Encryption/decryption failed.')
    return resbuf