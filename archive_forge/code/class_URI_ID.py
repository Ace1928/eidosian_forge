from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(init=False, slots=True)
class URI_ID:
    """
    An URI service ID.
    """
    protocol: bytes = attr.ib()
    dns_id: DNS_ID = attr.ib()
    pattern_class = URIPattern
    error_on_mismatch = URIMismatch

    def __init__(self, uri: str):
        if not isinstance(uri, str):
            raise TypeError('URI-ID must be a text string.')
        uri = uri.strip()
        if ':' not in uri or _is_ip_address(uri):
            raise ValueError('Invalid URI-ID.')
        prot, hostname = uri.split(':')
        self.protocol = prot.encode('ascii').translate(_TRANS_TO_LOWER)
        self.dns_id = DNS_ID(hostname.strip('/'))

    def verify(self, pattern: CertificatePattern) -> bool:
        """
        https://tools.ietf.org/search/rfc6125#section-6.5.2
        """
        if isinstance(pattern, self.pattern_class):
            return pattern.protocol_pattern == self.protocol and self.dns_id.verify(pattern.dns_pattern)
        return False