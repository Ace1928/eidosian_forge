from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(slots=True)
class IPAddress_ID:
    """
    An IP address service ID.
    """
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address = attr.ib(converter=ipaddress.ip_address)
    pattern_class = IPAddressPattern
    error_on_mismatch = IPAddressMismatch

    def verify(self, pattern: CertificatePattern) -> bool:
        """
        https://tools.ietf.org/search/rfc2818#section-3.1
        """
        if isinstance(pattern, self.pattern_class):
            return self.ip == pattern.pattern
        return False