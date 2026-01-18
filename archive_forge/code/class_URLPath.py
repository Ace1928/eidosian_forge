from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class URLPath(str):
    """
    A URL path string that may also hold an associated protocol and/or host.
    Used by the routing to return `url_path_for` matches.
    """

    def __new__(cls, path: str, protocol: str='', host: str='') -> 'URLPath':
        assert protocol in ('http', 'websocket', '')
        return str.__new__(cls, path)

    def __init__(self, path: str, protocol: str='', host: str='') -> None:
        self.protocol = protocol
        self.host = host

    def make_absolute_url(self, base_url: str | URL) -> URL:
        if isinstance(base_url, str):
            base_url = URL(base_url)
        if self.protocol:
            scheme = {'http': {True: 'https', False: 'http'}, 'websocket': {True: 'wss', False: 'ws'}}[self.protocol][base_url.is_secure]
        else:
            scheme = base_url.scheme
        netloc = self.host or base_url.netloc
        path = base_url.path.rstrip('/') + str(self)
        return URL(scheme=scheme, netloc=netloc, path=path)