import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union
from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate
def _decode_header_lines(lines: Iterable[bytes]) -> Iterable[Tuple[bytes, bytes]]:
    for line in _obsolete_line_fold(lines):
        matches = validate(header_field_re, line, 'illegal header line: {!r}', line)
        yield (matches['field_name'], matches['field_value'])