import typing
from starlette._exception_handler import (
from starlette.exceptions import HTTPException, WebSocketException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket
def add_exception_handler(self, exc_class_or_status_code: typing.Union[int, typing.Type[Exception]], handler: typing.Callable[[Request, Exception], Response]) -> None:
    if isinstance(exc_class_or_status_code, int):
        self._status_handlers[exc_class_or_status_code] = handler
    else:
        assert issubclass(exc_class_or_status_code, Exception)
        self._exception_handlers[exc_class_or_status_code] = handler