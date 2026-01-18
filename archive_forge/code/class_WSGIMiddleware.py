import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send
class WSGIMiddleware:

    def __init__(self, app: typing.Callable[..., typing.Any]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert scope['type'] == 'http'
        responder = WSGIResponder(self.app, scope)
        await responder(receive, send)