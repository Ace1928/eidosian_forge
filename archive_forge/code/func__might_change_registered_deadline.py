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
@contextmanager
@enable_ki_protection
def _might_change_registered_deadline(self) -> Iterator[None]:
    try:
        yield
    finally:
        old = self._registered_deadline
        if self._cancel_status is None or self._cancel_called:
            new = inf
        else:
            new = self._deadline
        if old != new:
            self._registered_deadline = new
            runner = GLOBAL_RUN_CONTEXT.runner
            if runner.is_guest:
                old_next_deadline = runner.deadlines.next_deadline()
            if old != inf:
                runner.deadlines.remove(old, self)
            if new != inf:
                runner.deadlines.add(new, self)
            if runner.is_guest:
                new_next_deadline = runner.deadlines.next_deadline()
                if old_next_deadline != new_next_deadline:
                    runner.force_guest_tick_asap()