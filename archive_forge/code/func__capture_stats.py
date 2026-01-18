import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
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