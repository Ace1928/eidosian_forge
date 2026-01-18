from __future__ import annotations
import io
from functools import partial
from typing import (
import trio
from ._util import async_wraps
from .abc import AsyncResource
class _CanRead1(Protocol):

    def read1(self, size: int | None=..., /) -> bytes:
        ...