import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union
from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate
def _obsolete_line_fold(lines: Iterable[bytes]) -> Iterable[bytes]:
    it = iter(lines)
    last: Optional[bytes] = None
    for line in it:
        match = obs_fold_re.match(line)
        if match:
            if last is None:
                raise LocalProtocolError('continuation line at start of headers')
            if not isinstance(last, bytearray):
                last = bytearray(last)
            last += b' '
            last += line[match.end():]
        else:
            if last is not None:
                yield last
            last = line
    if last is not None:
        yield last