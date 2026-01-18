from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _rsa_sig_setup(backend: Backend, padding: AsymmetricPadding, algorithm: typing.Optional[hashes.HashAlgorithm], key: typing.Union[_RSAPublicKey, _RSAPrivateKey], init_func: typing.Callable[[typing.Any], int]):
    padding_enum = _rsa_sig_determine_padding(backend, key, padding, algorithm)
    pkey_ctx = backend._lib.EVP_PKEY_CTX_new(key._evp_pkey, backend._ffi.NULL)
    backend.openssl_assert(pkey_ctx != backend._ffi.NULL)
    pkey_ctx = backend._ffi.gc(pkey_ctx, backend._lib.EVP_PKEY_CTX_free)
    res = init_func(pkey_ctx)
    if res != 1:
        errors = backend._consume_errors()
        raise ValueError('Unable to sign/verify with this key', errors)
    if algorithm is not None:
        evp_md = backend._evp_md_non_null_from_algorithm(algorithm)
        res = backend._lib.EVP_PKEY_CTX_set_signature_md(pkey_ctx, evp_md)
        if res <= 0:
            backend._consume_errors()
            raise UnsupportedAlgorithm('{} is not supported by this backend for RSA signing.'.format(algorithm.name), _Reasons.UNSUPPORTED_HASH)
    res = backend._lib.EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, padding_enum)
    if res <= 0:
        backend._consume_errors()
        raise UnsupportedAlgorithm('{} is not supported for the RSA signature operation.'.format(padding.name), _Reasons.UNSUPPORTED_PADDING)
    if isinstance(padding, PSS):
        assert isinstance(algorithm, hashes.HashAlgorithm)
        res = backend._lib.EVP_PKEY_CTX_set_rsa_pss_saltlen(pkey_ctx, _get_rsa_pss_salt_length(backend, padding, key, algorithm))
        backend.openssl_assert(res > 0)
        mgf1_md = backend._evp_md_non_null_from_algorithm(padding._mgf._algorithm)
        res = backend._lib.EVP_PKEY_CTX_set_rsa_mgf1_md(pkey_ctx, mgf1_md)
        backend.openssl_assert(res > 0)
    return pkey_ctx