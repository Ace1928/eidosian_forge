import ipaddress
from abc import ABCMeta
from typing import Any, cast, Dict, List, Optional, Union
import geoip2.records
from geoip2.mixins import SimpleEquality
class SimpleModel(SimpleEquality, metaclass=ABCMeta):
    """Provides basic methods for non-location models"""
    raw: Dict[str, Union[bool, str, int]]
    ip_address: str
    _network: Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]
    _prefix_len: int

    def __init__(self, raw: Dict[str, Union[bool, str, int]]) -> None:
        self.raw = raw
        self._network = None
        self._prefix_len = cast(int, raw.get('prefix_len'))
        self.ip_address = cast(str, raw.get('ip_address'))

    def __repr__(self) -> str:
        return f'{self.__module__}.{self.__class__.__name__}({self.raw})'

    @property
    def network(self) -> Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
        """The network for the record"""
        network = self._network
        if network is not None:
            return network
        ip_address = self.ip_address
        prefix_len = self._prefix_len
        if ip_address is None or prefix_len is None:
            return None
        network = ipaddress.ip_network(f'{ip_address}/{prefix_len}', False)
        self._network = network
        return network