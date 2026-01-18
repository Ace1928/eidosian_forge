from typing import Any, Callable, Generator, List
import pytest
from .._events import (
from .._headers import Headers, normalize_and_validate
from .._readers import (
from .._receivebuffer import ReceiveBuffer
from .._state import (
from .._util import LocalProtocolError
from .._writers import (
from .helpers import normalize_data_events
def _run_reader_iter(reader: Any, buf: bytes, do_eof: bool) -> Generator[Any, None, None]:
    while True:
        event = reader(buf)
        if event is None:
            break
        yield event
        if type(event) is EndOfMessage:
            break
    if do_eof:
        assert not buf
        yield reader.read_eof()