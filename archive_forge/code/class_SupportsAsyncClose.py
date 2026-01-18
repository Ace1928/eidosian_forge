from __future__ import annotations
import asyncio
import functools
import re
import sys
import typing
from contextlib import contextmanager
from starlette.types import Scope
class SupportsAsyncClose(typing.Protocol):

    async def close(self) -> None:
        ...