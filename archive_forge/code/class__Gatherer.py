import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
class _Gatherer(object):

    def __init__(self, submit_func, lock_factory, start_before_submit=False):
        self._submit_func = submit_func
        self._stats_lock = lock_factory()
        self._stats = ExecutorStatistics()
        self._start_before_submit = start_before_submit

    @property
    def statistics(self):
        return self._stats

    def clear(self):
        with self._stats_lock:
            self._stats = ExecutorStatistics()

    def _capture_stats(self, started_at, fut):
        """Capture statistics

        :param started_at: when the activity the future has performed
                           was started at
        :param fut: future object
        """
        elapsed = max(0.0, _utils.now() - started_at)
        with self._stats_lock:
            failures, executed, runtime, cancelled = (self._stats.failures, self._stats.executed, self._stats.runtime, self._stats.cancelled)
            if fut.cancelled():
                cancelled += 1
            else:
                executed += 1
                if fut.exception() is not None:
                    failures += 1
                runtime += elapsed
            self._stats = ExecutorStatistics(failures=failures, executed=executed, runtime=runtime, cancelled=cancelled)

    def submit(self, fn, *args, **kwargs):
        """Submit work to be executed and capture statistics."""
        if self._start_before_submit:
            started_at = _utils.now()
        fut = self._submit_func(fn, *args, **kwargs)
        if not self._start_before_submit:
            started_at = _utils.now()
        fut.add_done_callback(functools.partial(self._capture_stats, started_at))
        return fut