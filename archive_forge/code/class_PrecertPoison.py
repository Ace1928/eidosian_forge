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
class PrecertPoison(ExtensionType):
    oid = ExtensionOID.PRECERT_POISON

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecertPoison):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(PrecertPoison)

    def __repr__(self) -> str:
        return '<PrecertPoison()>'

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)