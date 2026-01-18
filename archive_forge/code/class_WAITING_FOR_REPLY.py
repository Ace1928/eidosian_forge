from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from ..core.types import ID
class WAITING_FOR_REPLY(State):
    """ The ``ClientConnection`` has sent a message to the Bokeh Server which
    should generate a paired reply, and is waiting for the reply.

    """
    _reply: Message[Any] | None

    def __init__(self, reqid: ID) -> None:
        self._reqid = reqid
        self._reply = None

    @property
    def reply(self) -> Message[Any] | None:
        """ The reply from the server. (``None`` until the reply arrives) """
        return self._reply

    @property
    def reqid(self) -> ID:
        """ The request ID of the originating message. """
        return self._reqid

    async def run(self, connection: ClientConnection) -> None:
        message = await connection._pop_message()
        if message is None:
            await connection._transition_to_disconnected(DISCONNECTED(ErrorReason.NETWORK_ERROR))
        elif 'reqid' in message.header and message.header['reqid'] == self.reqid:
            self._reply = message
            await connection._transition(CONNECTED_AFTER_ACK())
        else:
            await connection._next()