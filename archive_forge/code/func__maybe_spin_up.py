import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
def _maybe_spin_up(self):
    """Spin up a worker if needed."""
    if not self._workers or len(self._workers) < self._max_workers:
        w = _thread.ThreadWorker.create_and_register(self, self._work_queue)
        self._workers.append(w)
        w.start()