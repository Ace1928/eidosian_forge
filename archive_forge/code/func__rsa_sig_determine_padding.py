from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _rsa_sig_determine_padding(backend: Backend, key: typing.Union[_RSAPrivateKey, _RSAPublicKey], padding: AsymmetricPadding, algorithm: typing.Optional[hashes.HashAlgorithm]) -> int:
    if not isinstance(padding, AsymmetricPadding):
        raise TypeError('Expected provider of AsymmetricPadding.')
    pkey_size = backend._lib.EVP_PKEY_size(key._evp_pkey)
    backend.openssl_assert(pkey_size > 0)
    if isinstance(padding, PKCS1v15):
        padding_enum = backend._lib.RSA_PKCS1_PADDING
    elif isinstance(padding, PSS):
        if not isinstance(padding._mgf, MGF1):
            raise UnsupportedAlgorithm('Only MGF1 is supported by this backend.', _Reasons.UNSUPPORTED_MGF)
        if not isinstance(algorithm, hashes.HashAlgorithm):
            raise TypeError('Expected instance of hashes.HashAlgorithm.')
        if pkey_size - algorithm.digest_size - 2 < 0:
            raise ValueError('Digest too large for key size. Use a larger key or different digest.')
        padding_enum = backend._lib.RSA_PKCS1_PSS_PADDING
    else:
        raise UnsupportedAlgorithm(f'{padding.name} is not supported by this backend.', _Reasons.UNSUPPORTED_PADDING)
    return padding_enum