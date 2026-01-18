import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
from unittest import IsolatedAsyncioTestCase, mock
from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
import aiohttp
from aiohttp.client import _RequestContextManager, _WSRequestContextManager
from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .typedefs import StrOrURL
from .web import (
from .web_protocol import _RequestHandler
def _create_app_mock() -> mock.MagicMock:

    def get_dict(app: Any, key: str) -> Any:
        return app.__app_dict[key]

    def set_dict(app: Any, key: str, value: Any) -> None:
        app.__app_dict[key] = value
    app = mock.MagicMock(spec=Application)
    app.__app_dict = {}
    app.__getitem__ = get_dict
    app.__setitem__ = set_dict
    app._debug = False
    app.on_response_prepare = Signal(app)
    app.on_response_prepare.freeze()
    return app