import asyncio
import base64
import hashlib
import json
import os
import sys
import traceback
import warnings
from contextlib import suppress
from types import SimpleNamespace, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from yarl import URL
from . import hdrs, http, payload
from .abc import AbstractCookieJar
from .client_exceptions import (
from .client_reqrep import (
from .client_ws import ClientWebSocketResponse as ClientWebSocketResponse
from .connector import (
from .cookiejar import CookieJar
from .helpers import (
from .http import WS_KEY, HttpVersion, WebSocketReader, WebSocketWriter
from .http_websocket import WSHandshakeError, WSMessage, ws_ext_gen, ws_ext_parse
from .streams import FlowControlDataQueue
from .tracing import Trace, TraceConfig
from .typedefs import JSONEncoder, LooseCookies, LooseHeaders, StrOrURL
def _prepare_headers(self, headers: Optional[LooseHeaders]) -> 'CIMultiDict[str]':
    """Add default headers and transform it to CIMultiDict"""
    result = CIMultiDict(self._default_headers)
    if headers:
        if not isinstance(headers, (MultiDictProxy, MultiDict)):
            headers = CIMultiDict(headers)
        added_names: Set[str] = set()
        for key, value in headers.items():
            if key in added_names:
                result.add(key, value)
            else:
                result[key] = value
                added_names.add(key)
    return result