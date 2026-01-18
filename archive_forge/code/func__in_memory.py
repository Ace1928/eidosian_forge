from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
@property
def _in_memory(self) -> bool:
    rolled_to_disk = getattr(self.file, '_rolled', True)
    return not rolled_to_disk