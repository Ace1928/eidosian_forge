from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain
class NetworkRoute(BaseDomain):
    """Network Route Domain

    :param destination: str
           Destination network or host of this route.
    :param gateway: str
           Gateway for the route.
    """
    __slots__ = ('destination', 'gateway')

    def __init__(self, destination: str, gateway: str):
        self.destination = destination
        self.gateway = gateway