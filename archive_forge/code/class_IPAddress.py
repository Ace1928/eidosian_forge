from __future__ import annotations
import abc
import ipaddress
import typing
from email.utils import parseaddr
from cryptography.x509.name import Name
from cryptography.x509.oid import ObjectIdentifier
class IPAddress(GeneralName):

    def __init__(self, value: _IPAddressTypes) -> None:
        if not isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address, ipaddress.IPv4Network, ipaddress.IPv6Network)):
            raise TypeError('value must be an instance of ipaddress.IPv4Address, ipaddress.IPv6Address, ipaddress.IPv4Network, or ipaddress.IPv6Network')
        self._value = value

    @property
    def value(self) -> _IPAddressTypes:
        return self._value

    def _packed(self) -> bytes:
        if isinstance(self.value, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            return self.value.packed
        else:
            return self.value.network_address.packed + self.value.netmask.packed

    def __repr__(self) -> str:
        return f'<IPAddress(value={self.value})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IPAddress):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)