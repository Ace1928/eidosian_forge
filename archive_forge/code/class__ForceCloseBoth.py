from __future__ import annotations
import random
import sys
from contextlib import contextmanager, suppress
from typing import (
from .. import CancelScope, _core
from .._abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream, Stream
from .._highlevel_generic import aclose_forcefully
from ._checkpoints import assert_checkpoints
class _ForceCloseBoth(Generic[Res1, Res2]):

    def __init__(self, both: tuple[Res1, Res2]) -> None:
        self._first, self._second = both

    async def __aenter__(self) -> tuple[Res1, Res2]:
        return (self._first, self._second)

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        try:
            await aclose_forcefully(self._first)
        finally:
            await aclose_forcefully(self._second)