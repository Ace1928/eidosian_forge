from __future__ import annotations
import errno
import importlib.util
import os
import socket
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, NewType, Sequence
from urllib.parse import (
from urllib.parse import (
from urllib.request import pathname2url as _pathname2url
from _frozen_importlib_external import _NamespacePath
from jupyter_core.utils import ensure_async as _ensure_async
from packaging.version import Version
from tornado.httpclient import AsyncHTTPClient, HTTPClient, HTTPRequest, HTTPResponse
from tornado.netutil import Resolver
class UnixSocketResolver(Resolver):
    """A resolver that routes HTTP requests to unix sockets
            in tornado HTTP clients.
            Due to constraints in Tornados' API, the scheme of the
            must be `http` (not `http+unix`). Applications should replace
            the scheme in URLS before making a request to the HTTP client.
            """

    def initialize(self, resolver):
        self.resolver = resolver

    def close(self):
        self.resolver.close()

    async def resolve(self, host, port, *args, **kwargs):
        return [(socket.AF_UNIX, urldecode_unix_socket_path(host))]