from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
class _SingleResponse:

    def __init__(self, cert: x509.Certificate, issuer: x509.Certificate, algorithm: hashes.HashAlgorithm, cert_status: OCSPCertStatus, this_update: datetime.datetime, next_update: typing.Optional[datetime.datetime], revocation_time: typing.Optional[datetime.datetime], revocation_reason: typing.Optional[x509.ReasonFlags]):
        if not isinstance(cert, x509.Certificate) or not isinstance(issuer, x509.Certificate):
            raise TypeError('cert and issuer must be a Certificate')
        _verify_algorithm(algorithm)
        if not isinstance(this_update, datetime.datetime):
            raise TypeError('this_update must be a datetime object')
        if next_update is not None and (not isinstance(next_update, datetime.datetime)):
            raise TypeError('next_update must be a datetime object or None')
        self._cert = cert
        self._issuer = issuer
        self._algorithm = algorithm
        self._this_update = this_update
        self._next_update = next_update
        if not isinstance(cert_status, OCSPCertStatus):
            raise TypeError('cert_status must be an item from the OCSPCertStatus enum')
        if cert_status is not OCSPCertStatus.REVOKED:
            if revocation_time is not None:
                raise ValueError('revocation_time can only be provided if the certificate is revoked')
            if revocation_reason is not None:
                raise ValueError('revocation_reason can only be provided if the certificate is revoked')
        else:
            if not isinstance(revocation_time, datetime.datetime):
                raise TypeError('revocation_time must be a datetime object')
            revocation_time = _convert_to_naive_utc_time(revocation_time)
            if revocation_time < _EARLIEST_UTC_TIME:
                raise ValueError('The revocation_time must be on or after 1950 January 1.')
            if revocation_reason is not None and (not isinstance(revocation_reason, x509.ReasonFlags)):
                raise TypeError('revocation_reason must be an item from the ReasonFlags enum or None')
        self._cert_status = cert_status
        self._revocation_time = revocation_time
        self._revocation_reason = revocation_reason