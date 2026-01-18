from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class CreateImageResponse(BaseDomain):
    """Create Image Response Domain

    :param image: :class:`BoundImage <hcloud.images.client.BoundImage>`
           The Image which was created
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           The Action which shows the progress of the Floating IP Creation
    """
    __slots__ = ('action', 'image')

    def __init__(self, action: BoundAction, image: BoundImage):
        self.action = action
        self.image = image