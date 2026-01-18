from typing import Any, Callable, Dict, List, Tuple, Type, Union
from ._events import Data, EndOfMessage, Event, InformationalResponse, Request, Response
from ._headers import Headers
from ._state import CLIENT, IDLE, SEND_BODY, SEND_RESPONSE, SERVER
from ._util import LocalProtocolError, Sentinel
class ChunkedWriter(BodyWriter):

    def send_data(self, data: bytes, write: Writer) -> None:
        if not data:
            return
        write(b'%x\r\n' % len(data))
        write(data)
        write(b'\r\n')

    def send_eom(self, headers: Headers, write: Writer) -> None:
        write(b'0\r\n')
        write_headers(headers, write)