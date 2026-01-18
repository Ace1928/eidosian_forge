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
class OCSPNonce(ExtensionType):
    oid = OCSPExtensionOID.NONCE

    def __init__(self, nonce: bytes) -> None:
        if not isinstance(nonce, bytes):
            raise TypeError('nonce must be bytes')
        self._nonce = nonce

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OCSPNonce):
            return NotImplemented
        return self.nonce == other.nonce

    def __hash__(self) -> int:
        return hash(self.nonce)

    def __repr__(self) -> str:
        return f'<OCSPNonce(nonce={self.nonce!r})>'

    @property
    def nonce(self) -> bytes:
        return self._nonce

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)