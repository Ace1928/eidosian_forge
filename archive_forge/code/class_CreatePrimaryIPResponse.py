from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain
class CreatePrimaryIPResponse(BaseDomain):
    """Create Primary IP Response Domain

    :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`
           The Primary IP which was created
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           The Action which shows the progress of the Primary IP Creation
    """
    __slots__ = ('primary_ip', 'action')

    def __init__(self, primary_ip: BoundPrimaryIP, action: BoundAction | None):
        self.primary_ip = primary_ip
        self.action = action