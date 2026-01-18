from __future__ import annotations
import io
from typing import IO, TYPE_CHECKING, Any, Mapping, cast
from pip._vendor import msgpack
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3 import HTTPResponse
def _loads_v4(self, request: PreparedRequest, data: bytes, body_file: IO[bytes] | None=None) -> HTTPResponse | None:
    try:
        cached = msgpack.loads(data, raw=False)
    except ValueError:
        return None
    return self.prepare_response(request, cached, body_file)