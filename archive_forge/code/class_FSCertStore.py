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
class FSCertStore(CertStore):
    """File system certificate store."""

    def __init__(self, path: str) -> None:
        super().__init__()
        if os.path.isdir(path):
            path = os.path.join(path, '*')
        for p in glob.glob(path):
            with open(p) as f:
                cert = Certificate(f.read())
                if cert.has_expired():
                    raise SecurityError(f'Expired certificate: {cert.get_id()!r}')
                self.add_cert(cert)