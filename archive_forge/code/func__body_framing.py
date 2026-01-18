from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _body_framing(request_method: bytes, event: Union[Request, Response]) -> Tuple[str, Union[Tuple[()], Tuple[int]]]:
    assert type(event) in (Request, Response)
    if type(event) is Response:
        if event.status_code in (204, 304) or request_method == b'HEAD' or (request_method == b'CONNECT' and 200 <= event.status_code < 300):
            return ('content-length', (0,))
        assert event.status_code >= 200
    transfer_encodings = get_comma_header(event.headers, b'transfer-encoding')
    if transfer_encodings:
        assert transfer_encodings == [b'chunked']
        return ('chunked', ())
    content_lengths = get_comma_header(event.headers, b'content-length')
    if content_lengths:
        return ('content-length', (int(content_lengths[0]),))
    if type(event) is Request:
        return ('content-length', (0,))
    else:
        return ('http/1.0', ())