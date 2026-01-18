from collections import deque
from typing import (
import h11
from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .extensions import Extension
from .typing import Headers
from .utilities import (
def _process_connection_request(self, event: h11.Request) -> Request:
    if event.method != b'GET':
        raise RemoteProtocolError('Request method must be GET', event_hint=RejectConnection())
    connection_tokens = None
    extensions: List[str] = []
    host = None
    key = None
    subprotocols: List[str] = []
    upgrade = b''
    version = None
    headers: Headers = []
    for name, value in event.headers:
        name = name.lower()
        if name == b'connection':
            connection_tokens = split_comma_header(value)
        elif name == b'host':
            host = value.decode('idna')
            continue
        elif name == b'sec-websocket-extensions':
            extensions.extend(split_comma_header(value))
            continue
        elif name == b'sec-websocket-key':
            key = value
        elif name == b'sec-websocket-protocol':
            subprotocols.extend(split_comma_header(value))
            continue
        elif name == b'sec-websocket-version':
            version = value
        elif name == b'upgrade':
            upgrade = value
        headers.append((name, value))
    if connection_tokens is None or not any((token.lower() == 'upgrade' for token in connection_tokens)):
        raise RemoteProtocolError("Missing header, 'Connection: Upgrade'", event_hint=RejectConnection())
    if version != WEBSOCKET_VERSION:
        raise RemoteProtocolError("Missing header, 'Sec-WebSocket-Version'", event_hint=RejectConnection(headers=[(b'Sec-WebSocket-Version', WEBSOCKET_VERSION)], status_code=426 if version else 400))
    if key is None:
        raise RemoteProtocolError("Missing header, 'Sec-WebSocket-Key'", event_hint=RejectConnection())
    if upgrade.lower() != b'websocket':
        raise RemoteProtocolError("Missing header, 'Upgrade: WebSocket'", event_hint=RejectConnection())
    if host is None:
        raise RemoteProtocolError("Missing header, 'Host'", event_hint=RejectConnection())
    self._initiating_request = Request(extensions=extensions, extra_headers=headers, host=host, subprotocols=subprotocols, target=event.target.decode('ascii'))
    return self._initiating_request