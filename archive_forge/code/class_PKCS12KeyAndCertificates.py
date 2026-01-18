from __future__ import annotations
import typing
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives._serialization import PBES as PBES
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
class PKCS12KeyAndCertificates:

    def __init__(self, key: typing.Optional[PrivateKeyTypes], cert: typing.Optional[PKCS12Certificate], additional_certs: typing.List[PKCS12Certificate]):
        if key is not None and (not isinstance(key, (rsa.RSAPrivateKey, dsa.DSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey))):
            raise TypeError('Key must be RSA, DSA, EllipticCurve, ED25519, or ED448 private key, or None.')
        if cert is not None and (not isinstance(cert, PKCS12Certificate)):
            raise TypeError('cert must be a PKCS12Certificate object or None')
        if not all((isinstance(add_cert, PKCS12Certificate) for add_cert in additional_certs)):
            raise TypeError('all values in additional_certs must be PKCS12Certificate objects')
        self._key = key
        self._cert = cert
        self._additional_certs = additional_certs

    @property
    def key(self) -> typing.Optional[PrivateKeyTypes]:
        return self._key

    @property
    def cert(self) -> typing.Optional[PKCS12Certificate]:
        return self._cert

    @property
    def additional_certs(self) -> typing.List[PKCS12Certificate]:
        return self._additional_certs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PKCS12KeyAndCertificates):
            return NotImplemented
        return self.key == other.key and self.cert == other.cert and (self.additional_certs == other.additional_certs)

    def __hash__(self) -> int:
        return hash((self.key, self.cert, tuple(self.additional_certs)))

    def __repr__(self) -> str:
        fmt = '<PKCS12KeyAndCertificates(key={}, cert={}, additional_certs={})>'
        return fmt.format(self.key, self.cert, self.additional_certs)