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
class WebSocketServer:
    """
    WebSocket server returned by :func:`serve`.

    This class mirrors the API of :class:`~socketserver.BaseServer`, notably the
    :meth:`~socketserver.BaseServer.serve_forever` and
    :meth:`~socketserver.BaseServer.shutdown` methods, as well as the context
    manager protocol.

    Args:
        socket: Server socket listening for new connections.
        handler: Handler for one connection. Receives the socket and address
            returned by :meth:`~socket.socket.accept`.
        logger: Logger for this server.

    """

    def __init__(self, socket: socket.socket, handler: Callable[[socket.socket, Any], None], logger: Optional[LoggerLike]=None):
        self.socket = socket
        self.handler = handler
        if logger is None:
            logger = logging.getLogger('websockets.server')
        self.logger = logger
        if sys.platform != 'win32':
            self.shutdown_watcher, self.shutdown_notifier = os.pipe()

    def serve_forever(self) -> None:
        """
        See :meth:`socketserver.BaseServer.serve_forever`.

        This method doesn't return. Calling :meth:`shutdown` from another thread
        stops the server.

        Typical use::

            with serve(...) as server:
                server.serve_forever()

        """
        poller = selectors.DefaultSelector()
        poller.register(self.socket, selectors.EVENT_READ)
        if sys.platform != 'win32':
            poller.register(self.shutdown_watcher, selectors.EVENT_READ)
        while True:
            poller.select()
            try:
                sock, addr = self.socket.accept()
            except OSError:
                break
            thread = threading.Thread(target=self.handler, args=(sock, addr))
            thread.start()

    def shutdown(self) -> None:
        """
        See :meth:`socketserver.BaseServer.shutdown`.

        """
        self.socket.close()
        if sys.platform != 'win32':
            os.write(self.shutdown_notifier, b'x')

    def fileno(self) -> int:
        """
        See :meth:`socketserver.BaseServer.fileno`.

        """
        return self.socket.fileno()

    def __enter__(self) -> WebSocketServer:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.shutdown()