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
@final
@attrs.define(eq=False, repr=False)
class CancelScope:
    """A *cancellation scope*: the link between a unit of cancellable
    work and Trio's cancellation system.

    A :class:`CancelScope` becomes associated with some cancellable work
    when it is used as a context manager surrounding that work::

        cancel_scope = trio.CancelScope()
        ...
        with cancel_scope:
            await long_running_operation()

    Inside the ``with`` block, a cancellation of ``cancel_scope`` (via
    a call to its :meth:`cancel` method or via the expiry of its
    :attr:`deadline`) will immediately interrupt the
    ``long_running_operation()`` by raising :exc:`Cancelled` at its
    next :ref:`checkpoint <checkpoints>`.

    The context manager ``__enter__`` returns the :class:`CancelScope`
    object itself, so you can also write ``with trio.CancelScope() as
    cancel_scope:``.

    If a cancel scope becomes cancelled before entering its ``with`` block,
    the :exc:`Cancelled` exception will be raised at the first
    checkpoint inside the ``with`` block. This allows a
    :class:`CancelScope` to be created in one :ref:`task <tasks>` and
    passed to another, so that the first task can later cancel some work
    inside the second.

    Cancel scopes are not reusable or reentrant; that is, each cancel
    scope can be used for at most one ``with`` block.  (You'll get a
    :exc:`RuntimeError` if you violate this rule.)

    The :class:`CancelScope` constructor takes initial values for the
    cancel scope's :attr:`deadline` and :attr:`shield` attributes; these
    may be freely modified after construction, whether or not the scope
    has been entered yet, and changes take immediate effect.
    """
    _cancel_status: CancelStatus | None = attrs.field(default=None, init=False)
    _has_been_entered: bool = attrs.field(default=False, init=False)
    _registered_deadline: float = attrs.field(default=inf, init=False)
    _cancel_called: bool = attrs.field(default=False, init=False)
    cancelled_caught: bool = attrs.field(default=False, init=False)
    _deadline: float = attrs.field(default=inf, kw_only=True, alias='deadline')
    _shield: bool = attrs.field(default=False, kw_only=True, alias='shield')

    @enable_ki_protection
    def __enter__(self) -> Self:
        task = _core.current_task()
        if self._has_been_entered:
            raise RuntimeError("Each CancelScope may only be used for a single 'with' block")
        self._has_been_entered = True
        if current_time() >= self._deadline:
            self.cancel()
        with self._might_change_registered_deadline():
            self._cancel_status = CancelStatus(scope=self, parent=task._cancel_status)
            task._activate_cancel_status(self._cancel_status)
        return self

    def _close(self, exc: BaseException | None) -> BaseException | None:
        if self._cancel_status is None:
            new_exc = RuntimeError(f'Cancel scope stack corrupted: attempted to exit {self!r} which had already been exited')
            new_exc.__context__ = exc
            return new_exc
        scope_task = current_task()
        if scope_task._cancel_status is not self._cancel_status:
            if self._cancel_status.abandoned_by_misnesting:
                pass
            elif not self._cancel_status.encloses(scope_task._cancel_status):
                new_exc = RuntimeError(f'Cancel scope stack corrupted: attempted to exit {self!r} from unrelated {scope_task!r}\n{MISNESTING_ADVICE}')
                new_exc.__context__ = exc
                return new_exc
            else:
                new_exc = RuntimeError(f"Cancel scope stack corrupted: attempted to exit {self!r} in {scope_task!r} that's still within its child {scope_task._cancel_status._scope!r}\n{MISNESTING_ADVICE}")
                new_exc.__context__ = exc
                exc = new_exc
                scope_task._activate_cancel_status(self._cancel_status.parent)
        else:
            scope_task._activate_cancel_status(self._cancel_status.parent)
        if exc is not None and self._cancel_status.effectively_cancelled and (not self._cancel_status.parent_cancellation_is_visible_to_us):
            if isinstance(exc, Cancelled):
                self.cancelled_caught = True
                exc = None
            elif isinstance(exc, BaseExceptionGroup):
                matched, exc = exc.split(Cancelled)
                if matched:
                    self.cancelled_caught = True
                if exc:
                    exc = collapse_exception_group(exc)
        self._cancel_status.close()
        with self._might_change_registered_deadline():
            self._cancel_status = None
        return exc

    def __exit__(self, etype: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
        remaining_error_after_cancel_scope = self._close(exc)
        if remaining_error_after_cancel_scope is None:
            return True
        elif remaining_error_after_cancel_scope is exc:
            return False
        else:
            old_context = remaining_error_after_cancel_scope.__context__
            try:
                raise remaining_error_after_cancel_scope
            finally:
                _, value, _ = sys.exc_info()
                assert value is remaining_error_after_cancel_scope
                value.__context__ = old_context
                del remaining_error_after_cancel_scope, value, _, exc
                locals()

    def __repr__(self) -> str:
        if self._cancel_status is not None:
            binding = 'active'
        elif self._has_been_entered:
            binding = 'exited'
        else:
            binding = 'unbound'
        if self._cancel_called:
            state = ', cancelled'
        elif self._deadline == inf:
            state = ''
        else:
            try:
                now = current_time()
            except RuntimeError:
                state = ''
            else:
                state = ', deadline is {:.2f} seconds {}'.format(abs(self._deadline - now), 'from now' if self._deadline >= now else 'ago')
        return f'<trio.CancelScope at {id(self):#x}, {binding}{state}>'

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

    @property
    def deadline(self) -> float:
        """Read-write, :class:`float`. An absolute time on the current
        run's clock at which this scope will automatically become
        cancelled. You can adjust the deadline by modifying this
        attribute, e.g.::

           # I need a little more time!
           cancel_scope.deadline += 30

        Note that for efficiency, the core run loop only checks for
        expired deadlines every once in a while. This means that in
        certain cases there may be a short delay between when the clock
        says the deadline should have expired, and when checkpoints
        start raising :exc:`~trio.Cancelled`. This is a very obscure
        corner case that you're unlikely to notice, but we document it
        for completeness. (If this *does* cause problems for you, of
        course, then `we want to know!
        <https://github.com/python-trio/trio/issues>`__)

        Defaults to :data:`math.inf`, which means "no deadline", though
        this can be overridden by the ``deadline=`` argument to
        the :class:`~trio.CancelScope` constructor.
        """
        return self._deadline

    @deadline.setter
    def deadline(self, new_deadline: float) -> None:
        with self._might_change_registered_deadline():
            self._deadline = float(new_deadline)

    @property
    def shield(self) -> bool:
        """Read-write, :class:`bool`, default :data:`False`. So long as
        this is set to :data:`True`, then the code inside this scope
        will not receive :exc:`~trio.Cancelled` exceptions from scopes
        that are outside this scope. They can still receive
        :exc:`~trio.Cancelled` exceptions from (1) this scope, or (2)
        scopes inside this scope. You can modify this attribute::

           with trio.CancelScope() as cancel_scope:
               cancel_scope.shield = True
               # This cannot be interrupted by any means short of
               # killing the process:
               await sleep(10)

               cancel_scope.shield = False
               # Now this can be cancelled normally:
               await sleep(10)

        Defaults to :data:`False`, though this can be overridden by the
        ``shield=`` argument to the :class:`~trio.CancelScope` constructor.
        """
        return self._shield

    @shield.setter
    @enable_ki_protection
    def shield(self, new_value: bool) -> None:
        if not isinstance(new_value, bool):
            raise TypeError('shield must be a bool')
        self._shield = new_value
        if self._cancel_status is not None:
            self._cancel_status.recalculate()

    @enable_ki_protection
    def cancel(self) -> None:
        """Cancels this scope immediately.

        This method is idempotent, i.e., if the scope was already
        cancelled then this method silently does nothing.
        """
        if self._cancel_called:
            return
        with self._might_change_registered_deadline():
            self._cancel_called = True
        if self._cancel_status is not None:
            self._cancel_status.recalculate()

    @property
    def cancel_called(self) -> bool:
        """Readonly :class:`bool`. Records whether cancellation has been
        requested for this scope, either by an explicit call to
        :meth:`cancel` or by the deadline expiring.

        This attribute being True does *not* necessarily mean that the
        code within the scope has been, or will be, affected by the
        cancellation. For example, if :meth:`cancel` was called after
        the last checkpoint in the ``with`` block, when it's too late to
        deliver a :exc:`~trio.Cancelled` exception, then this attribute
        will still be True.

        This attribute is mostly useful for debugging and introspection.
        If you want to know whether or not a chunk of code was actually
        cancelled, then :attr:`cancelled_caught` is usually more
        appropriate.
        """
        if self._cancel_status is not None or not self._has_been_entered:
            if not self._cancel_called and current_time() >= self._deadline:
                self.cancel()
        return self._cancel_called