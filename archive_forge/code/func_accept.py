from __future__ import annotations
import base64
import binascii
import email.utils
import http
import warnings
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, cast
from .datastructures import Headers, MultipleValuesError
from .exceptions import (
from .extensions import Extension, ServerExtensionFactory
from .headers import (
from .http11 import Request, Response
from .protocol import CONNECTING, OPEN, SERVER, Protocol, State
from .typing import (
from .utils import accept_key
from .legacy.server import *  # isort:skip  # noqa: I001
from .legacy.server import __all__ as legacy__all__
def accept(self, request: Request) -> Response:
    """
        Create a handshake response to accept the connection.

        If the connection cannot be established, the handshake response
        actually rejects the handshake.

        You must send the handshake response with :meth:`send_response`.

        You may modify it before sending it, for example to add HTTP headers.

        Args:
            request: WebSocket handshake request event received from the client.

        Returns:
            WebSocket handshake response event to send to the client.

        """
    try:
        accept_header, extensions_header, protocol_header = self.process_request(request)
    except InvalidOrigin as exc:
        request._exception = exc
        self.handshake_exc = exc
        if self.debug:
            self.logger.debug('! invalid origin', exc_info=True)
        return self.reject(http.HTTPStatus.FORBIDDEN, f'Failed to open a WebSocket connection: {exc}.\n')
    except InvalidUpgrade as exc:
        request._exception = exc
        self.handshake_exc = exc
        if self.debug:
            self.logger.debug('! invalid upgrade', exc_info=True)
        response = self.reject(http.HTTPStatus.UPGRADE_REQUIRED, f'Failed to open a WebSocket connection: {exc}.\n\nYou cannot access a WebSocket server directly with a browser. You need a WebSocket client.\n')
        response.headers['Upgrade'] = 'websocket'
        return response
    except InvalidHandshake as exc:
        request._exception = exc
        self.handshake_exc = exc
        if self.debug:
            self.logger.debug('! invalid handshake', exc_info=True)
        return self.reject(http.HTTPStatus.BAD_REQUEST, f'Failed to open a WebSocket connection: {exc}.\n')
    except Exception as exc:
        request._exception = exc
        self.handshake_exc = exc
        self.logger.error('opening handshake failed', exc_info=True)
        return self.reject(http.HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to open a WebSocket connection.\nSee server log for more information.\n')
    headers = Headers()
    headers['Date'] = email.utils.formatdate(usegmt=True)
    headers['Upgrade'] = 'websocket'
    headers['Connection'] = 'Upgrade'
    headers['Sec-WebSocket-Accept'] = accept_header
    if extensions_header is not None:
        headers['Sec-WebSocket-Extensions'] = extensions_header
    if protocol_header is not None:
        headers['Sec-WebSocket-Protocol'] = protocol_header
    self.logger.info('connection open')
    return Response(101, 'Switching Protocols', headers)