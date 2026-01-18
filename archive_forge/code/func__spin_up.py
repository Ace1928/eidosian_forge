import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
def _spin_up(self, work):
    """Spin up a greenworker if less than max_workers.

        :param work: work to be given to the greenworker
        :returns: whether a green worker was spun up or not
        :rtype: boolean
        """
    alive = self._pool.running() + self._pool.waiting()
    if alive < self._max_workers:
        self._pool.spawn_n(_green.GreenWorker(work, self._delayed_work))
        return True
    return False