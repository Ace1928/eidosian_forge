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
class ExtendedKeyUsage(ExtensionType):
    oid = ExtensionOID.EXTENDED_KEY_USAGE

    def __init__(self, usages: typing.Iterable[ObjectIdentifier]) -> None:
        usages = list(usages)
        if not all((isinstance(x, ObjectIdentifier) for x in usages)):
            raise TypeError('Every item in the usages list must be an ObjectIdentifier')
        self._usages = usages
    __len__, __iter__, __getitem__ = _make_sequence_methods('_usages')

    def __repr__(self) -> str:
        return f'<ExtendedKeyUsage({self._usages})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtendedKeyUsage):
            return NotImplemented
        return self._usages == other._usages

    def __hash__(self) -> int:
        return hash(tuple(self._usages))

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)