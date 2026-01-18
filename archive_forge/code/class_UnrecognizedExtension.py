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
class UnrecognizedExtension(ExtensionType):

    def __init__(self, oid: ObjectIdentifier, value: bytes) -> None:
        if not isinstance(oid, ObjectIdentifier):
            raise TypeError('oid must be an ObjectIdentifier')
        self._oid = oid
        self._value = value

    @property
    def oid(self) -> ObjectIdentifier:
        return self._oid

    @property
    def value(self) -> bytes:
        return self._value

    def __repr__(self) -> str:
        return '<UnrecognizedExtension(oid={0.oid}, value={0.value!r})>'.format(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnrecognizedExtension):
            return NotImplemented
        return self.oid == other.oid and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.oid, self.value))

    def public_bytes(self) -> bytes:
        return self.value