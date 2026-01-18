from typing import Any, Callable, Dict, List, Tuple, Type, Union
from ._events import Data, EndOfMessage, Event, InformationalResponse, Request, Response
from ._headers import Headers
from ._state import CLIENT, IDLE, SEND_BODY, SEND_RESPONSE, SERVER
from ._util import LocalProtocolError, Sentinel
class ContentLengthWriter(BodyWriter):

    def __init__(self, length: int) -> None:
        self._length = length

    def send_data(self, data: bytes, write: Writer) -> None:
        self._length -= len(data)
        if self._length < 0:
            raise LocalProtocolError('Too much data for declared Content-Length')
        write(data)

    def send_eom(self, headers: Headers, write: Writer) -> None:
        if self._length != 0:
            raise LocalProtocolError('Too little data for declared Content-Length')
        if headers:
            raise LocalProtocolError("Content-Length and trailers don't mix")