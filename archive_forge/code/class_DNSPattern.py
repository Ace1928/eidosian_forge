from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(slots=True)
class DNSPattern:
    """
    A DNS pattern as extracted from certificates.
    """
    pattern: bytes = attr.ib()
    _RE_LEGAL_CHARS = re.compile(b'^[a-z0-9\\-_.]+$')

    @classmethod
    def from_bytes(cls, pattern: bytes) -> DNSPattern:
        if not isinstance(pattern, bytes):
            raise TypeError('The DNS pattern must be a bytes string.')
        pattern = pattern.strip()
        if pattern == b'' or _is_ip_address(pattern) or b'\x00' in pattern:
            raise CertificateError(f'Invalid DNS pattern {pattern!r}.')
        pattern = pattern.translate(_TRANS_TO_LOWER)
        if b'*' in pattern:
            _validate_pattern(pattern)
        return cls(pattern=pattern)