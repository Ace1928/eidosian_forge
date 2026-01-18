from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class ResetPasswordResponse(BaseDomain):
    """Reset Password Response Domain

    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the server passwort reset action
    :param root_password: str
           The root password of the server
    """
    __slots__ = ('action', 'root_password')

    def __init__(self, action: BoundAction, root_password: str):
        self.action = action
        self.root_password = root_password