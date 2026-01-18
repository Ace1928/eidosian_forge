from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
def _validate_pattern(cert_pattern: bytes) -> None:
    """
    Check whether the usage of wildcards within *cert_pattern* conforms with
    our expectations.
    """
    cnt = cert_pattern.count(b'*')
    if cnt > 1:
        raise CertificateError(f"Certificate's DNS-ID {cert_pattern!r} contains too many wildcards.")
    parts = cert_pattern.split(b'.')
    if len(parts) < 3:
        raise CertificateError(f"Certificate's DNS-ID {cert_pattern!r} has too few host components for wildcard usage.")
    if b'*' not in parts[0]:
        raise CertificateError("Certificate's DNS-ID {!r} has a wildcard outside the left-most part.".format(cert_pattern))
    if any((not len(p) for p in parts)):
        raise CertificateError(f"Certificate's DNS-ID {cert_pattern!r} contains empty parts.")