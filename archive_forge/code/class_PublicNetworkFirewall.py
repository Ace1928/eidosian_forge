from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class PublicNetworkFirewall(BaseDomain):
    """Public Network Domain

    :param firewall: :class:`BoundFirewall <hcloud.firewalls.domain.BoundFirewall>`
    :param status: str
    """
    __slots__ = ('firewall', 'status')
    STATUS_APPLIED = 'applied'
    'Public Network Firewall Status applied'
    STATUS_PENDING = 'pending'
    'Public Network Firewall Status pending'

    def __init__(self, firewall: BoundFirewall, status: str):
        self.firewall = firewall
        self.status = status