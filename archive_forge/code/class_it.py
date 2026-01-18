from __future__ import annotations
import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
import outcome
import pytest
import sniffio
from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD
from .tutil import (
class it:

    def __init__(self, count: int) -> None:
        self.count = count
        self.val = 0

    async def __anext__(self) -> int:
        await sleep(0)
        val = self.val
        if val >= self.count:
            raise StopAsyncIteration
        self.val += 1
        return val