from __future__ import annotations
import io
from functools import partial
from typing import (
import trio
from ._util import async_wraps
from .abc import AsyncResource
class _CanReadInto1(Protocol):

    def readinto1(self, buffer: Buffer, /) -> int:
        ...