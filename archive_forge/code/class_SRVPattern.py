from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(slots=True)
class SRVPattern:
    """
    An SRV pattern as extracted from certificates.
    """
    name_pattern: bytes = attr.ib()
    dns_pattern: DNSPattern = attr.ib()

    @classmethod
    def from_bytes(cls, pattern: bytes) -> SRVPattern:
        if not isinstance(pattern, bytes):
            raise TypeError('The SRV pattern must be a bytes string.')
        pattern = pattern.strip().translate(_TRANS_TO_LOWER)
        if pattern[0] != b'_'[0] or b'.' not in pattern or b'*' in pattern or _is_ip_address(pattern):
            raise CertificateError(f'Invalid SRV pattern {pattern!r}.')
        name, hostname = pattern.split(b'.', 1)
        return cls(name_pattern=name[1:], dns_pattern=DNSPattern.from_bytes(hostname))