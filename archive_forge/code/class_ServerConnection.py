from __future__ import annotations
import http
import logging
import os
import selectors
import socket
import ssl
import sys
import threading
from types import TracebackType
from typing import Any, Callable, Optional, Sequence, Type
from websockets.frames import CloseCode
from ..extensions.base import ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import validate_subprotocols
from ..http import USER_AGENT
from ..http11 import Request, Response
from ..protocol import CONNECTING, OPEN, Event
from ..server import ServerProtocol
from ..typing import LoggerLike, Origin, Subprotocol
from .connection import Connection
from .utils import Deadline
class ServerConnection(Connection):
    """
    Threaded implementation of a WebSocket server connection.

    :class:`ServerConnection` provides :meth:`recv` and :meth:`send` methods for
    receiving and sending messages.

    It supports iteration to receive messages::

        for message in websocket:
            process(message)

    The iterator exits normally when the connection is closed with close code
    1000 (OK) or 1001 (going away) or without a close code. It raises a
    :exc:`~websockets.exceptions.ConnectionClosedError` when the connection is
    closed with any other code.

    Args:
        socket: Socket connected to a WebSocket client.
        protocol: Sans-I/O connection.
        close_timeout: Timeout for closing the connection in seconds.

    """

    def __init__(self, socket: socket.socket, protocol: ServerProtocol, *, close_timeout: Optional[float]=10) -> None:
        self.protocol: ServerProtocol
        self.request_rcvd = threading.Event()
        super().__init__(socket, protocol, close_timeout=close_timeout)

    def handshake(self, process_request: Optional[Callable[[ServerConnection, Request], Optional[Response]]]=None, process_response: Optional[Callable[[ServerConnection, Request, Response], Optional[Response]]]=None, server_header: Optional[str]=USER_AGENT, timeout: Optional[float]=None) -> None:
        """
        Perform the opening handshake.

        """
        if not self.request_rcvd.wait(timeout):
            self.close_socket()
            self.recv_events_thread.join()
            raise TimeoutError('timed out during handshake')
        if self.request is None:
            self.close_socket()
            self.recv_events_thread.join()
            raise ConnectionError('connection closed during handshake')
        with self.send_context(expected_state=CONNECTING):
            self.response = None
            if process_request is not None:
                try:
                    self.response = process_request(self, self.request)
                except Exception as exc:
                    self.protocol.handshake_exc = exc
                    self.logger.error('opening handshake failed', exc_info=True)
                    self.response = self.protocol.reject(http.HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to open a WebSocket connection.\nSee server log for more information.\n')
            if self.response is None:
                self.response = self.protocol.accept(self.request)
            if server_header is not None:
                self.response.headers['Server'] = server_header
            if process_response is not None:
                try:
                    response = process_response(self, self.request, self.response)
                except Exception as exc:
                    self.protocol.handshake_exc = exc
                    self.logger.error('opening handshake failed', exc_info=True)
                    self.response = self.protocol.reject(http.HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to open a WebSocket connection.\nSee server log for more information.\n')
                else:
                    if response is not None:
                        self.response = response
            self.protocol.send_response(self.response)
        if self.protocol.state is not OPEN:
            self.recv_events_thread.join(self.close_timeout)
            self.close_socket()
            self.recv_events_thread.join()
        if self.protocol.handshake_exc is not None:
            raise self.protocol.handshake_exc

    def process_event(self, event: Event) -> None:
        """
        Process one incoming event.

        """
        if self.request is None:
            assert isinstance(event, Request)
            self.request = event
            self.request_rcvd.set()
        else:
            super().process_event(event)

    def recv_events(self) -> None:
        """
        Read incoming data from the socket and process events.

        """
        try:
            super().recv_events()
        finally:
            self.request_rcvd.set()