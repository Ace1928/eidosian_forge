from __future__ import annotations
import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar
import trio
from ._core import (
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error
@attrs.frozen(eq=False, slots=False)
class RunSync(Generic[RetT]):
    fn: Callable[..., RetT]
    args: tuple[object, ...]
    context: contextvars.Context = attrs.field(init=False, factory=contextvars.copy_context)
    queue: stdlib_queue.SimpleQueue[outcome.Outcome[RetT]] = attrs.field(init=False, factory=stdlib_queue.SimpleQueue)

    @disable_ki_protection
    def unprotected_fn(self) -> RetT:
        ret = self.context.run(self.fn, *self.args)
        if inspect.iscoroutine(ret):
            ret.close()
            raise TypeError('Trio expected a synchronous function, but {!r} appears to be asynchronous'.format(getattr(self.fn, '__qualname__', self.fn)))
        return ret

    def run_sync(self) -> None:
        result = outcome.capture(self.unprotected_fn)
        self.queue.put_nowait(result)

    def run_in_host_task(self, token: TrioToken) -> None:
        task_register = PARENT_TASK_DATA.task_register

        def in_trio_thread() -> None:
            task = task_register[0]
            assert task is not None, 'guaranteed by abandon_on_cancel semantics'
            trio.lowlevel.reschedule(task, outcome.Value(self))
        token.run_sync_soon(in_trio_thread)

    def run_in_system_nursery(self, token: TrioToken) -> None:
        token.run_sync_soon(self.run_sync)