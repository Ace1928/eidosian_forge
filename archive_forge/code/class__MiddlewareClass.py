from __future__ import annotations
import sys
from typing import Any, Iterator, Protocol
from starlette.types import ASGIApp, Receive, Scope, Send
class _MiddlewareClass(Protocol[P]):

    def __init__(self, app: ASGIApp, *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        ...