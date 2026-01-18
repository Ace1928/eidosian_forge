from __future__ import annotations
import abc
import datetime
import hashlib
import ipaddress
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import asn1
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.certificate_transparency import (
from cryptography.x509.general_name import (
from cryptography.x509.name import Name, RelativeDistinguishedName
from cryptography.x509.oid import (
def _key_identifier_from_public_key(public_key: CertificatePublicKeyTypes) -> bytes:
    if isinstance(public_key, RSAPublicKey):
        data = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.PKCS1)
    elif isinstance(public_key, EllipticCurvePublicKey):
        data = public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    else:
        serialized = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        data = asn1.parse_spki_for_data(serialized)
    return hashlib.sha1(data).digest()