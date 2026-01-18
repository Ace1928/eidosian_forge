from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class CreateServerResponse(BaseDomain):
    """Create Server Response Domain

    :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>`
           The created server
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the server creation
    :param next_actions: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
           Additional actions like a `start_server` action after the server creation
    :param root_password: str, None
           The root password of the server if no SSH-Key was given on server creation
    """
    __slots__ = ('server', 'action', 'next_actions', 'root_password')

    def __init__(self, server: BoundServer, action: BoundAction, next_actions: list[BoundAction], root_password: str | None):
        self.server = server
        self.action = action
        self.next_actions = next_actions
        self.root_password = root_password