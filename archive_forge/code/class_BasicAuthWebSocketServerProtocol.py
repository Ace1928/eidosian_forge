from __future__ import annotations
import functools
import hmac
import http
from typing import Any, Awaitable, Callable, Iterable, Optional, Tuple, Union, cast
from ..datastructures import Headers
from ..exceptions import InvalidHeader
from ..headers import build_www_authenticate_basic, parse_authorization_basic
from .server import HTTPResponse, WebSocketServerProtocol
class BasicAuthWebSocketServerProtocol(WebSocketServerProtocol):
    """
    WebSocket server protocol that enforces HTTP Basic Auth.

    """
    realm: str = ''
    '\n    Scope of protection.\n\n    If provided, it should contain only ASCII characters because the\n    encoding of non-ASCII characters is undefined.\n    '
    username: Optional[str] = None
    'Username of the authenticated user.'

    def __init__(self, *args: Any, realm: Optional[str]=None, check_credentials: Optional[Callable[[str, str], Awaitable[bool]]]=None, **kwargs: Any) -> None:
        if realm is not None:
            self.realm = realm
        self._check_credentials = check_credentials
        super().__init__(*args, **kwargs)

    async def check_credentials(self, username: str, password: str) -> bool:
        """
        Check whether credentials are authorized.

        This coroutine may be overridden in a subclass, for example to
        authenticate against a database or an external service.

        Args:
            username: HTTP Basic Auth username.
            password: HTTP Basic Auth password.

        Returns:
            bool: :obj:`True` if the handshake should continue;
            :obj:`False` if it should fail with an HTTP 401 error.

        """
        if self._check_credentials is not None:
            return await self._check_credentials(username, password)
        return False

    async def process_request(self, path: str, request_headers: Headers) -> Optional[HTTPResponse]:
        """
        Check HTTP Basic Auth and return an HTTP 401 response if needed.

        """
        try:
            authorization = request_headers['Authorization']
        except KeyError:
            return (http.HTTPStatus.UNAUTHORIZED, [('WWW-Authenticate', build_www_authenticate_basic(self.realm))], b'Missing credentials\n')
        try:
            username, password = parse_authorization_basic(authorization)
        except InvalidHeader:
            return (http.HTTPStatus.UNAUTHORIZED, [('WWW-Authenticate', build_www_authenticate_basic(self.realm))], b'Unsupported credentials\n')
        if not await self.check_credentials(username, password):
            return (http.HTTPStatus.UNAUTHORIZED, [('WWW-Authenticate', build_www_authenticate_basic(self.realm))], b'Invalid credentials\n')
        self.username = username
        return await super().process_request(path, request_headers)