from __future__ import annotations
import contextlib
import operator as _operator
import ssl as _stdlib_ssl
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Any, ClassVar, Final as TFinal, Generic, TypeVar
import trio
from . import _sync
from ._highlevel_generic import aclose_forcefully
from ._util import ConflictDetector, final
from .abc import Listener, Stream
class _Once:

    def __init__(self, afn: Callable[..., Awaitable[object]], *args: object) -> None:
        self._afn = afn
        self._args = args
        self.started = False
        self._done = _sync.Event()

    async def ensure(self, *, checkpoint: bool) -> None:
        if not self.started:
            self.started = True
            await self._afn(*self._args)
            self._done.set()
        elif not checkpoint and self._done.is_set():
            return
        else:
            await self._done.wait()

    @property
    def done(self) -> bool:
        return bool(self._done.is_set())