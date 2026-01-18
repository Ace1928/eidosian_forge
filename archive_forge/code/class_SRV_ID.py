from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(init=False, slots=True)
class SRV_ID:
    """
    An SRV service ID.
    """
    name: bytes = attr.ib()
    dns_id: DNS_ID = attr.ib()
    pattern_class = SRVPattern
    error_on_mismatch = SRVMismatch

    def __init__(self, srv: str):
        if not isinstance(srv, str):
            raise TypeError('SRV-ID must be a text string.')
        srv = srv.strip()
        if '.' not in srv or _is_ip_address(srv) or srv[0] != '_':
            raise ValueError('Invalid SRV-ID.')
        name, hostname = srv.split('.', 1)
        self.name = name[1:].encode('ascii').translate(_TRANS_TO_LOWER)
        self.dns_id = DNS_ID(hostname)

    def verify(self, pattern: CertificatePattern) -> bool:
        """
        https://tools.ietf.org/search/rfc6125#section-6.5.1
        """
        if isinstance(pattern, self.pattern_class):
            return self.name == pattern.name_pattern and self.dns_id.verify(pattern.dns_pattern)
        return False