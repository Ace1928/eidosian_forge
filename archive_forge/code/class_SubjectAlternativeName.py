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
class SubjectAlternativeName(ExtensionType):
    oid = ExtensionOID.SUBJECT_ALTERNATIVE_NAME

    def __init__(self, general_names: typing.Iterable[GeneralName]) -> None:
        self._general_names = GeneralNames(general_names)
    __len__, __iter__, __getitem__ = _make_sequence_methods('_general_names')

    @typing.overload
    def get_values_for_type(self, type: typing.Union[typing.Type[DNSName], typing.Type[UniformResourceIdentifier], typing.Type[RFC822Name]]) -> typing.List[str]:
        ...

    @typing.overload
    def get_values_for_type(self, type: typing.Type[DirectoryName]) -> typing.List[Name]:
        ...

    @typing.overload
    def get_values_for_type(self, type: typing.Type[RegisteredID]) -> typing.List[ObjectIdentifier]:
        ...

    @typing.overload
    def get_values_for_type(self, type: typing.Type[IPAddress]) -> typing.List[_IPAddressTypes]:
        ...

    @typing.overload
    def get_values_for_type(self, type: typing.Type[OtherName]) -> typing.List[OtherName]:
        ...

    def get_values_for_type(self, type: typing.Union[typing.Type[DNSName], typing.Type[DirectoryName], typing.Type[IPAddress], typing.Type[OtherName], typing.Type[RFC822Name], typing.Type[RegisteredID], typing.Type[UniformResourceIdentifier]]) -> typing.Union[typing.List[_IPAddressTypes], typing.List[str], typing.List[OtherName], typing.List[Name], typing.List[ObjectIdentifier]]:
        return self._general_names.get_values_for_type(type)

    def __repr__(self) -> str:
        return f'<SubjectAlternativeName({self._general_names})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubjectAlternativeName):
            return NotImplemented
        return self._general_names == other._general_names

    def __hash__(self) -> int:
        return hash(self._general_names)

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)