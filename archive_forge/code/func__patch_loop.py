import asyncio
import asyncio.events as events
import os
import sys
import threading
from contextlib import contextmanager, suppress
from heapq import heappop
def _patch_loop(loop):
    """Patch loop to make it reentrant."""

    def run_forever(self):
        with manage_run(self), manage_asyncgens(self):
            while True:
                self._run_once()
                if self._stopping:
                    break
        self._stopping = False

    def run_until_complete(self, future):
        with manage_run(self):
            f = asyncio.ensure_future(future, loop=self)
            if f is not future:
                f._log_destroy_pending = False
            while not f.done():
                self._run_once()
                if self._stopping:
                    break
            if not f.done():
                raise RuntimeError('Event loop stopped before Future completed.')
            return f.result()

    def _run_once(self):
        """
        Simplified re-implementation of asyncio's _run_once that
        runs handles as they become ready.
        """
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)
        timeout = 0 if ready or self._stopping else min(max(scheduled[0]._when - self.time(), 0), 86400) if scheduled else None
        event_list = self._selector.select(timeout)
        self._process_events(event_list)
        end_time = self.time() + self._clock_resolution
        while scheduled and scheduled[0]._when < end_time:
            handle = heappop(scheduled)
            ready.append(handle)
        for _ in range(len(ready)):
            if not ready:
                break
            handle = ready.popleft()
            if not handle._cancelled:
                curr_task = curr_tasks.pop(self, None)
                try:
                    handle._run()
                finally:
                    if curr_task is not None:
                        curr_tasks[self] = curr_task
        handle = None

    @contextmanager
    def manage_run(self):
        """Set up the loop for running."""
        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = events._get_running_loop()
        try:
            self._thread_id = threading.get_ident()
            events._set_running_loop(self)
            self._num_runs_pending += 1
            if self._is_proactorloop:
                if self._self_reading_future is None:
                    self.call_soon(self._loop_self_reading)
            yield
        finally:
            self._thread_id = old_thread_id
            events._set_running_loop(old_running_loop)
            self._num_runs_pending -= 1
            if self._is_proactorloop:
                if self._num_runs_pending == 0 and self._self_reading_future is not None:
                    ov = self._self_reading_future._ov
                    self._self_reading_future.cancel()
                    if ov is not None:
                        self._proactor._unregister(ov)
                    self._self_reading_future = None

    @contextmanager
    def manage_asyncgens(self):
        if not hasattr(sys, 'get_asyncgen_hooks'):
            return
        old_agen_hooks = sys.get_asyncgen_hooks()
        try:
            self._set_coroutine_origin_tracking(self._debug)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(firstiter=self._asyncgen_firstiter_hook, finalizer=self._asyncgen_finalizer_hook)
            yield
        finally:
            self._set_coroutine_origin_tracking(False)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(*old_agen_hooks)

    def _check_running(self):
        """Do not throw exception if loop is already running."""
        pass
    if hasattr(loop, '_nest_patched'):
        return
    if not isinstance(loop, asyncio.BaseEventLoop):
        raise ValueError("Can't patch loop of type %s" % type(loop))
    cls = loop.__class__
    cls.run_forever = run_forever
    cls.run_until_complete = run_until_complete
    cls._run_once = _run_once
    cls._check_running = _check_running
    cls._check_runnung = _check_running
    cls._num_runs_pending = 1 if loop.is_running() else 0
    cls._is_proactorloop = os.name == 'nt' and issubclass(cls, asyncio.ProactorEventLoop)
    if sys.version_info < (3, 7, 0):
        cls._set_coroutine_origin_tracking = cls._set_coroutine_wrapper
    curr_tasks = asyncio.tasks._current_tasks if sys.version_info >= (3, 7, 0) else asyncio.Task._current_tasks
    cls._nest_patched = True