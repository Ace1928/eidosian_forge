from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _ecdsa_sig_verify(backend: Backend, public_key: _EllipticCurvePublicKey, signature: bytes, data: bytes) -> None:
    res = backend._lib.ECDSA_verify(0, data, len(data), signature, len(signature), public_key._ec_key)
    if res != 1:
        backend._consume_errors()
        raise InvalidSignature