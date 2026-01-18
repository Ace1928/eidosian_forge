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
class AuthorityInformationAccess(ExtensionType):
    oid = ExtensionOID.AUTHORITY_INFORMATION_ACCESS

    def __init__(self, descriptions: typing.Iterable[AccessDescription]) -> None:
        descriptions = list(descriptions)
        if not all((isinstance(x, AccessDescription) for x in descriptions)):
            raise TypeError('Every item in the descriptions list must be an AccessDescription')
        self._descriptions = descriptions
    __len__, __iter__, __getitem__ = _make_sequence_methods('_descriptions')

    def __repr__(self) -> str:
        return f'<AuthorityInformationAccess({self._descriptions})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AuthorityInformationAccess):
            return NotImplemented
        return self._descriptions == other._descriptions

    def __hash__(self) -> int:
        return hash(tuple(self._descriptions))

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)