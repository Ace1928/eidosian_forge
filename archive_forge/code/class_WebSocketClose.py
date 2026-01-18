from __future__ import annotations
import enum
import json
import typing
from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send
class WebSocketClose:

    def __init__(self, code: int=1000, reason: str | None=None) -> None:
        self.code = code
        self.reason = reason or ''

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send({'type': 'websocket.close', 'code': self.code, 'reason': self.reason})