from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
def _coordinateThisTask(self, task: Callable[..., object]) -> None:
    """
        Select a worker to dispatch to, either an idle one or a new one, and
        perform it.

        This method should run on the coordinator worker.

        @param task: the task to dispatch
        @type task: 0-argument callable
        """
    worker = self._idle.pop() if self._idle else self._createWorker()
    if worker is None:
        self._pending.append(task)
        return
    not_none_worker = worker
    self._busyCount += 1

    @worker.do
    def doWork() -> None:
        try:
            task()
        except BaseException:
            self._logException()

        @self._coordinator.do
        def idleAndPending() -> None:
            self._busyCount -= 1
            self._recycleWorker(not_none_worker)