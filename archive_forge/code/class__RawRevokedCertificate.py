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
class _RawRevokedCertificate(RevokedCertificate):

    def __init__(self, serial_number: int, revocation_date: datetime.datetime, extensions: Extensions):
        self._serial_number = serial_number
        self._revocation_date = revocation_date
        self._extensions = extensions

    @property
    def serial_number(self) -> int:
        return self._serial_number

    @property
    def revocation_date(self) -> datetime.datetime:
        return self._revocation_date

    @property
    def extensions(self) -> Extensions:
        return self._extensions