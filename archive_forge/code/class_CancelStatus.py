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
@attrs.define(eq=False)
class CancelStatus:
    """Tracks the cancellation status for a contiguous extent
    of code that will become cancelled, or not, as a unit.

    Each task has at all times a single "active" CancelStatus whose
    cancellation state determines whether checkpoints executed in that
    task raise Cancelled. Each 'with CancelScope(...)' context is
    associated with a particular CancelStatus.  When a task enters
    such a context, a CancelStatus is created which becomes the active
    CancelStatus for that task; when the 'with' block is exited, the
    active CancelStatus for that task goes back to whatever it was
    before.

    CancelStatus objects are arranged in a tree whose structure
    mirrors the lexical nesting of the cancel scope contexts.  When a
    CancelStatus becomes cancelled, it notifies all of its direct
    children, who become cancelled in turn (and continue propagating
    the cancellation down the tree) unless they are shielded. (There
    will be at most one such child except in the case of a
    CancelStatus that immediately encloses a nursery.) At the leaves
    of this tree are the tasks themselves, which get woken up to deliver
    an abort when their direct parent CancelStatus becomes cancelled.

    You can think of CancelStatus as being responsible for the
    "plumbing" of cancellations as oppposed to CancelScope which is
    responsible for the origination of them.

    """
    _scope: CancelScope = attrs.field(alias='scope')
    effectively_cancelled: bool = False
    _parent: CancelStatus | None = attrs.field(default=None, repr=False, alias='parent')
    _children: set[CancelStatus] = attrs.field(factory=set, init=False, repr=False)
    _tasks: set[Task] = attrs.field(factory=set, init=False, repr=False)
    abandoned_by_misnesting: bool = attrs.field(default=False, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._parent is not None:
            self._parent._children.add(self)
            self.recalculate()

    @property
    def parent(self) -> CancelStatus | None:
        return self._parent

    @parent.setter
    def parent(self, parent: CancelStatus) -> None:
        if self._parent is not None:
            self._parent._children.remove(self)
        self._parent = parent
        if self._parent is not None:
            self._parent._children.add(self)
            self.recalculate()

    @property
    def children(self) -> frozenset[CancelStatus]:
        return frozenset(self._children)

    @property
    def tasks(self) -> frozenset[Task]:
        return frozenset(self._tasks)

    def encloses(self, other: CancelStatus | None) -> bool:
        """Returns true if this cancel status is a direct or indirect
        parent of cancel status *other*, or if *other* is *self*.
        """
        while other is not None:
            if other is self:
                return True
            other = other.parent
        return False

    def close(self) -> None:
        self.parent = None
        if self._tasks or self._children:
            self._mark_abandoned()
            self.effectively_cancelled = True
            for task in self._tasks:
                task._attempt_delivery_of_any_pending_cancel()
            for child in self._children:
                child.recalculate()

    @property
    def parent_cancellation_is_visible_to_us(self) -> bool:
        return self._parent is not None and (not self._scope.shield) and self._parent.effectively_cancelled

    def recalculate(self) -> None:
        todo = [self]
        while todo:
            current = todo.pop()
            new_state = current._scope.cancel_called or current.parent_cancellation_is_visible_to_us
            if new_state != current.effectively_cancelled:
                current.effectively_cancelled = new_state
                if new_state:
                    for task in current._tasks:
                        task._attempt_delivery_of_any_pending_cancel()
                todo.extend(current._children)

    def _mark_abandoned(self) -> None:
        self.abandoned_by_misnesting = True
        for child in self._children:
            child._mark_abandoned()

    def effective_deadline(self) -> float:
        if self.effectively_cancelled:
            return -inf
        if self._parent is None or self._scope.shield:
            return self._scope.deadline
        return min(self._scope.deadline, self._parent.effective_deadline())