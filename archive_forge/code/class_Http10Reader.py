import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union
from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate
class Http10Reader:

    def __call__(self, buf: ReceiveBuffer) -> Optional[Data]:
        data = buf.maybe_extract_at_most(999999999)
        if data is None:
            return None
        return Data(data=data)

    def read_eof(self) -> EndOfMessage:
        return EndOfMessage()