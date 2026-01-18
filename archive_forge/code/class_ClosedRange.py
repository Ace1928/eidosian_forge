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
class ClosedRange(NamedTuple):
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __bool__(self) -> bool:
        return len(self) > 0