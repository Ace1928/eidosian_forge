from __future__ import annotations
import enum
import functools
import gc
import itertools
import random
import select
import sys
import threading
import warnings
from collections import deque
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from contextvars import copy_context
from heapq import heapify, heappop, heappush
from math import inf
from time import perf_counter
from typing import (
import attrs
from outcome import Error, Outcome, Value, capture
from sniffio import thread_local as sniffio_library
from sortedcontainers import SortedDict
from .. import _core
from .._abc import Clock, Instrument
from .._deprecate import warn_deprecated
from .._util import NoPublicConstructor, coroutine_or_error, final
from ._asyncgens import AsyncGenerators
from ._concat_tb import concat_tb
from ._entry_queue import EntryQueue, TrioToken
from ._exceptions import Cancelled, RunFinishedError, TrioInternalError
from ._instrumentation import Instruments
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED, KIManager, enable_ki_protection
from ._thread_cache import start_thread_soon
from ._traps import (
from ._generated_instrumentation import *
from ._generated_run import *
@attrs.define(slots=False)
class NurseryManager:
    """Nursery context manager.

    Note we explicitly avoid @asynccontextmanager and @async_generator
    since they add a lot of extraneous stack frames to exceptions, as
    well as cause problematic behavior with handling of StopIteration
    and StopAsyncIteration.

    """
    strict_exception_groups: bool = True

    @enable_ki_protection
    async def __aenter__(self) -> Nursery:
        self._scope = CancelScope()
        self._scope.__enter__()
        self._nursery = Nursery._create(current_task(), self._scope, self.strict_exception_groups)
        return self._nursery

    @enable_ki_protection
    async def __aexit__(self, etype: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        new_exc = await self._nursery._nested_child_finished(exc)
        combined_error_from_nursery = self._scope._close(new_exc)
        if combined_error_from_nursery is None:
            return True
        elif combined_error_from_nursery is exc:
            return False
        else:
            old_context = combined_error_from_nursery.__context__
            try:
                raise combined_error_from_nursery
            finally:
                _, value, _ = sys.exc_info()
                assert value is combined_error_from_nursery
                value.__context__ = old_context
                del _, combined_error_from_nursery, value, new_exc
    if not TYPE_CHECKING:

        def __enter__(self) -> NoReturn:
            raise RuntimeError("use 'async with open_nursery(...)', not 'with open_nursery(...)'")

        def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> NoReturn:
            raise AssertionError('Never called, but should be defined')