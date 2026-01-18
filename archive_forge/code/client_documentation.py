from __future__ import annotations
import asyncio
import functools
import logging
import random
import urllib.parse
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike
from ..exceptions import (
from ..extensions import ClientExtensionFactory, Extension
from ..extensions.permessage_deflate import enable_client_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..typing import ExtensionHeader, LoggerLike, Origin, Subprotocol
from ..uri import WebSocketURI, parse_uri
from .compatibility import asyncio_timeout
from .handshake import build_request, check_response
from .http import read_response
from .protocol import WebSocketCommonProtocol

        Perform the client side of the opening handshake.

        Args:
            wsuri: URI of the WebSocket server.
            origin: Value of the ``Origin`` header.
            extensions: List of supported extensions, in order in which they
                should be negotiated and run.
            subprotocols: List of supported subprotocols, in order of decreasing
                preference.
            extra_headers: Arbitrary HTTP headers to add to the handshake request.

        Raises:
            InvalidHandshake: If the handshake fails.

        