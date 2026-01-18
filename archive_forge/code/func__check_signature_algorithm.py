from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _check_signature_algorithm(signature_algorithm: ec.EllipticCurveSignatureAlgorithm) -> None:
    if not isinstance(signature_algorithm, ec.ECDSA):
        raise UnsupportedAlgorithm('Unsupported elliptic curve signature algorithm.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)