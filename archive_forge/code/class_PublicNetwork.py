from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class PublicNetwork(BaseDomain):
    """Public Network Domain

    :param ipv4: :class:`IPv4Address <hcloud.load_balancers.domain.IPv4Address>`
    :param ipv6: :class:`IPv6Network <hcloud.load_balancers.domain.IPv6Network>`
    :param enabled:  boolean
    """
    __slots__ = ('ipv4', 'ipv6', 'enabled')

    def __init__(self, ipv4: IPv4Address, ipv6: IPv6Network, enabled: bool):
        self.ipv4 = ipv4
        self.ipv6 = ipv6
        self.enabled = enabled