from __future__ import annotations
import functools
import inspect
import sys
import typing
from urllib.parse import urlencode
from starlette._utils import is_async_callable
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection, Request
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocket
class AuthenticationBackend:

    async def authenticate(self, conn: HTTPConnection) -> tuple[AuthCredentials, BaseUser] | None:
        raise NotImplementedError()