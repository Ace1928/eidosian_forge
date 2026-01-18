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
class PrecertificateSignedCertificateTimestamps(ExtensionType):
    oid = ExtensionOID.PRECERT_SIGNED_CERTIFICATE_TIMESTAMPS

    def __init__(self, signed_certificate_timestamps: typing.Iterable[SignedCertificateTimestamp]) -> None:
        signed_certificate_timestamps = list(signed_certificate_timestamps)
        if not all((isinstance(sct, SignedCertificateTimestamp) for sct in signed_certificate_timestamps)):
            raise TypeError('Every item in the signed_certificate_timestamps list must be a SignedCertificateTimestamp')
        self._signed_certificate_timestamps = signed_certificate_timestamps
    __len__, __iter__, __getitem__ = _make_sequence_methods('_signed_certificate_timestamps')

    def __repr__(self) -> str:
        return '<PrecertificateSignedCertificateTimestamps({})>'.format(list(self))

    def __hash__(self) -> int:
        return hash(tuple(self._signed_certificate_timestamps))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecertificateSignedCertificateTimestamps):
            return NotImplemented
        return self._signed_certificate_timestamps == other._signed_certificate_timestamps

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)