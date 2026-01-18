from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
class CertificateRevocationListBuilder:
    _extensions: typing.List[Extension[ExtensionType]]
    _revoked_certificates: typing.List[RevokedCertificate]

    def __init__(self, issuer_name: typing.Optional[Name]=None, last_update: typing.Optional[datetime.datetime]=None, next_update: typing.Optional[datetime.datetime]=None, extensions: typing.List[Extension[ExtensionType]]=[], revoked_certificates: typing.List[RevokedCertificate]=[]):
        self._issuer_name = issuer_name
        self._last_update = last_update
        self._next_update = next_update
        self._extensions = extensions
        self._revoked_certificates = revoked_certificates

    def issuer_name(self, issuer_name: Name) -> CertificateRevocationListBuilder:
        if not isinstance(issuer_name, Name):
            raise TypeError('Expecting x509.Name object.')
        if self._issuer_name is not None:
            raise ValueError('The issuer name may only be set once.')
        return CertificateRevocationListBuilder(issuer_name, self._last_update, self._next_update, self._extensions, self._revoked_certificates)

    def last_update(self, last_update: datetime.datetime) -> CertificateRevocationListBuilder:
        if not isinstance(last_update, datetime.datetime):
            raise TypeError('Expecting datetime object.')
        if self._last_update is not None:
            raise ValueError('Last update may only be set once.')
        last_update = _convert_to_naive_utc_time(last_update)
        if last_update < _EARLIEST_UTC_TIME:
            raise ValueError('The last update date must be on or after 1950 January 1.')
        if self._next_update is not None and last_update > self._next_update:
            raise ValueError('The last update date must be before the next update date.')
        return CertificateRevocationListBuilder(self._issuer_name, last_update, self._next_update, self._extensions, self._revoked_certificates)

    def next_update(self, next_update: datetime.datetime) -> CertificateRevocationListBuilder:
        if not isinstance(next_update, datetime.datetime):
            raise TypeError('Expecting datetime object.')
        if self._next_update is not None:
            raise ValueError('Last update may only be set once.')
        next_update = _convert_to_naive_utc_time(next_update)
        if next_update < _EARLIEST_UTC_TIME:
            raise ValueError('The last update date must be on or after 1950 January 1.')
        if self._last_update is not None and next_update < self._last_update:
            raise ValueError('The next update date must be after the last update date.')
        return CertificateRevocationListBuilder(self._issuer_name, self._last_update, next_update, self._extensions, self._revoked_certificates)

    def add_extension(self, extval: ExtensionType, critical: bool) -> CertificateRevocationListBuilder:
        """
        Adds an X.509 extension to the certificate revocation list.
        """
        if not isinstance(extval, ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return CertificateRevocationListBuilder(self._issuer_name, self._last_update, self._next_update, self._extensions + [extension], self._revoked_certificates)

    def add_revoked_certificate(self, revoked_certificate: RevokedCertificate) -> CertificateRevocationListBuilder:
        """
        Adds a revoked certificate to the CRL.
        """
        if not isinstance(revoked_certificate, RevokedCertificate):
            raise TypeError('Must be an instance of RevokedCertificate')
        return CertificateRevocationListBuilder(self._issuer_name, self._last_update, self._next_update, self._extensions, self._revoked_certificates + [revoked_certificate])

    def sign(self, private_key: CertificateIssuerPrivateKeyTypes, algorithm: typing.Optional[_AllowedHashTypes], backend: typing.Any=None) -> CertificateRevocationList:
        if self._issuer_name is None:
            raise ValueError('A CRL must have an issuer name')
        if self._last_update is None:
            raise ValueError('A CRL must have a last update time')
        if self._next_update is None:
            raise ValueError('A CRL must have a next update time')
        return rust_x509.create_x509_crl(self, private_key, algorithm)