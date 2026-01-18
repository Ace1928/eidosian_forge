from __future__ import annotations
import os
import re
import stat
from typing import NamedTuple
from urllib.parse import quote
import aiofiles
from aiofiles.os import stat as aio_stat
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response, guess_type
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
class RangedStaticFiles(StaticFiles):

    def file_response(self, full_path: str | os.PathLike, stat_result: os.stat_result, scope: Scope, status_code: int=200) -> Response:
        request_headers = Headers(scope=scope)
        if request_headers.get('range'):
            response = self.ranged_file_response(full_path, stat_result=stat_result, scope=scope)
        else:
            response = super().file_response(full_path, stat_result=stat_result, scope=scope, status_code=status_code)
        response.headers['accept-ranges'] = 'bytes'
        return response

    def ranged_file_response(self, full_path: str | os.PathLike, stat_result: os.stat_result, scope: Scope) -> Response:
        method = scope['method']
        request_headers = Headers(scope=scope)
        range_header = request_headers['range']
        match = RANGE_REGEX.search(range_header)
        if not match:
            raise HTTPException(400)
        start, end = (match.group('start'), match.group('end'))
        range = OpenRange(int(start), int(end) if end else None)
        return RangedFileResponse(full_path, range, stat_result=stat_result, method=method)