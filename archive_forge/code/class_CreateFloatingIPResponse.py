from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain
class CreateFloatingIPResponse(BaseDomain):
    """Create Floating IP Response Domain

    :param floating_ip: :class:`BoundFloatingIP <hcloud.floating_ips.client.BoundFloatingIP>`
           The Floating IP which was created
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           The Action which shows the progress of the Floating IP Creation
    """
    __slots__ = ('floating_ip', 'action')

    def __init__(self, floating_ip: BoundFloatingIP, action: BoundAction | None):
        self.floating_ip = floating_ip
        self.action = action