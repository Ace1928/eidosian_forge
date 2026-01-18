from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
def _recycleWorker(self, worker: IWorker) -> None:
    """
        Called only from coordinator.

        Recycle the given worker into the idle pool.

        @param worker: a worker created by C{createWorker} and now idle.
        @type worker: L{IWorker}
        """
    self._idle.add(worker)
    if self._pending:
        self._coordinateThisTask(self._pending.popleft())
    elif self._shouldQuitCoordinator:
        self._quitIdlers()
    elif self._toShrink > 0:
        self._toShrink -= 1
        self._idle.remove(worker)
        worker.quit()