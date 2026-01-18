from __future__ import annotations
import asyncio
import functools
import re
import sys
import typing
from contextlib import contextmanager
from starlette.types import Scope
class AwaitableOrContextManagerWrapper(typing.Generic[SupportsAsyncCloseType]):
    __slots__ = ('aw', 'entered')

    def __init__(self, aw: typing.Awaitable[SupportsAsyncCloseType]) -> None:
        self.aw = aw

    def __await__(self) -> typing.Generator[typing.Any, None, SupportsAsyncCloseType]:
        return self.aw.__await__()

    async def __aenter__(self) -> SupportsAsyncCloseType:
        self.entered = await self.aw
        return self.entered

    async def __aexit__(self, *args: typing.Any) -> None | bool:
        await self.entered.close()
        return None