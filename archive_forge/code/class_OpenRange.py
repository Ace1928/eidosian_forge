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
class OpenRange(NamedTuple):
    start: int
    end: int | None = None

    def clamp(self, start: int, end: int) -> ClosedRange:
        begin = max(self.start, start)
        end = min((x for x in (self.end, end) if x))
        begin = min(begin, end)
        end = max(begin, end)
        return ClosedRange(begin, end)