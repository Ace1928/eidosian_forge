from __future__ import annotations
import datetime
import glob
import os
from typing import TYPE_CHECKING, Iterator
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509 import load_pem_x509_certificate
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from celery.exceptions import SecurityError
from .utils import reraise_errors
class CertStore:
    """Base class for certificate stores."""

    def __init__(self) -> None:
        self._certs: dict[str, Certificate] = {}

    def itercerts(self) -> Iterator[Certificate]:
        """Return certificate iterator."""
        yield from self._certs.values()

    def __getitem__(self, id: str) -> Certificate:
        """Get certificate by id."""
        try:
            return self._certs[bytes_to_str(id)]
        except KeyError:
            raise SecurityError(f'Unknown certificate: {id!r}')

    def add_cert(self, cert: Certificate) -> None:
        cert_id = bytes_to_str(cert.get_id())
        if cert_id in self._certs:
            raise SecurityError(f'Duplicate certificate: {id!r}')
        self._certs[cert_id] = cert