from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ..core import BaseDomain
class FirewallResourceAppliedToResources(BaseDomain):
    """Firewall Resource applied to Domain

    :param type: Type of resource referenced
    :param server: Server the Firewall is applied to
    """
    __slots__ = ('type', 'server')

    def __init__(self, type: str, server: BoundServer | None=None):
        self.type = type
        self.server = server