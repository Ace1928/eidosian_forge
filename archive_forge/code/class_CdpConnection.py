import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
class CdpConnection(CdpBase, trio.abc.AsyncResource):
    """Contains the connection state for a Chrome DevTools Protocol server.

    CDP can multiplex multiple "sessions" over a single connection. This
    class corresponds to the "root" session, i.e. the implicitly created
    session that has no session ID. This class is responsible for
    reading incoming WebSocket messages and forwarding them to the
    corresponding session, as well as handling messages targeted at the
    root session itself. You should generally call the
    :func:`open_cdp()` instead of instantiating this class directly.
    """

    def __init__(self, ws):
        """Constructor.

        :param trio_websocket.WebSocketConnection ws:
        """
        super().__init__(ws, session_id=None, target_id=None)
        self.sessions = {}

    async def aclose(self):
        """Close the underlying WebSocket connection.

        This will cause the reader task to gracefully exit when it tries
        to read the next message from the WebSocket. All of the public
        APIs (``execute()``, ``listen()``, etc.) will raise
        ``CdpConnectionClosed`` after the CDP connection is closed. It
        is safe to call this multiple times.
        """
        await self.ws.aclose()

    @asynccontextmanager
    async def open_session(self, target_id) -> typing.AsyncIterator[CdpSession]:
        """This context manager opens a session and enables the "simple" style
        of calling CDP APIs.

        For example, inside a session context, you can call ``await
        dom.get_document()`` and it will execute on the current session
        automatically.
        """
        session = await self.connect_session(target_id)
        with session_context(session):
            yield session

    async def connect_session(self, target_id) -> 'CdpSession':
        """Returns a new :class:`CdpSession` connected to the specified
        target."""
        global devtools
        session_id = await self.execute(devtools.target.attach_to_target(target_id, True))
        session = CdpSession(self.ws, session_id, target_id)
        self.sessions[session_id] = session
        return session

    async def _reader_task(self):
        """Runs in the background and handles incoming messages: dispatching
        responses to commands and events to listeners."""
        global devtools
        while True:
            try:
                message = await self.ws.get_message()
            except WsConnectionClosed:
                break
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                raise BrowserError({'code': -32700, 'message': 'Client received invalid JSON', 'data': message})
            logger.debug('Received message %r', data)
            if 'sessionId' in data:
                session_id = devtools.target.SessionID(data['sessionId'])
                try:
                    session = self.sessions[session_id]
                except KeyError:
                    raise BrowserError({'code': -32700, 'message': 'Browser sent a message for an invalid session', 'data': f'{session_id!r}'})
                session._handle_data(data)
            else:
                self._handle_data(data)
        for _, session in self.sessions.items():
            for _, senders in session.channels.items():
                for sender in senders:
                    sender.close()