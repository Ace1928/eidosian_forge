from __future__ import annotations
import logging # isort:skip
from typing import Any, Awaitable, Callable
from tornado import locks
from tornado.websocket import WebSocketClientConnection
class WebSocketClientConnectionWrapper:
    """ Used for compatibility across Tornado versions and to add write_lock"""

    def __init__(self, socket: WebSocketClientConnection) -> None:
        self._socket = socket
        self.write_lock = locks.Lock()

    async def write_message(self, message: str | bytes, binary: bool=False, locked: bool=True) -> None:
        """ Write a message to the websocket after obtaining the appropriate
        Bokeh Document lock.

        """
        if locked:
            with await self.write_lock.acquire():
                self._socket.write_message(message, binary)
        else:
            self._socket.write_message(message, binary)

    def close(self, code: int | None=None, reason: str | None=None) -> None:
        """ Close the websocket. """
        return self._socket.close(code, reason)

    def read_message(self, callback: Callable[..., Any] | None=None) -> Awaitable[None | str | bytes]:
        """ Read a message from websocket and execute a callback.

        """
        return self._socket.read_message(callback)