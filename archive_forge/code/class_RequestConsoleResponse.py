from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class RequestConsoleResponse(BaseDomain):
    """Request Console Response Domain

    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the server request console action
    :param wss_url: str
           URL of websocket proxy to use. This includes a token which is valid for a limited time only.
    :param password: str
           VNC password to use for this connection. This password only works in combination with a wss_url with valid token.
    """
    __slots__ = ('action', 'wss_url', 'password')

    def __init__(self, action: BoundAction, wss_url: str, password: str):
        self.action = action
        self.wss_url = wss_url
        self.password = password