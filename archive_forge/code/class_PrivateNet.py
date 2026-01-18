from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class PrivateNet(BaseDomain):
    """PrivateNet Domain

    :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>`
           The Network the LoadBalancer is attached to
    :param ip: str
           The main IP Address of the LoadBalancer in the Network
    """
    __slots__ = ('network', 'ip')

    def __init__(self, network: BoundNetwork, ip: str):
        self.network = network
        self.ip = ip