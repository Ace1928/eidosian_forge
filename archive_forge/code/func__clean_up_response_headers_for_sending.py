from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _clean_up_response_headers_for_sending(self, response: Response) -> Response:
    assert type(response) is Response
    headers = response.headers
    need_close = False
    method_for_choosing_headers = cast(bytes, self._request_method)
    if method_for_choosing_headers == b'HEAD':
        method_for_choosing_headers = b'GET'
    framing_type, _ = _body_framing(method_for_choosing_headers, response)
    if framing_type in ('chunked', 'http/1.0'):
        headers = set_comma_header(headers, b'content-length', [])
        if self.their_http_version is None or self.their_http_version < b'1.1':
            headers = set_comma_header(headers, b'transfer-encoding', [])
            if self._request_method != b'HEAD':
                need_close = True
        else:
            headers = set_comma_header(headers, b'transfer-encoding', [b'chunked'])
    if not self._cstate.keep_alive or need_close:
        connection = set(get_comma_header(headers, b'connection'))
        connection.discard(b'keep-alive')
        connection.add(b'close')
        headers = set_comma_header(headers, b'connection', sorted(connection))
    return Response(headers=headers, status_code=response.status_code, http_version=response.http_version, reason=response.reason)