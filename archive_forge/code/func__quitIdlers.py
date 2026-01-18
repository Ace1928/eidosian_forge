from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
def _quitIdlers(self, n: Optional[int]=None) -> None:
    """
        The implmentation of C{shrink}, performed by the coordinator worker.

        @param n: see L{Team.shrink}
        """
    if n is None:
        n = len(self._idle) + self._busyCount
    for x in range(n):
        if self._idle:
            self._idle.pop().quit()
        else:
            self._toShrink += 1
    if self._shouldQuitCoordinator and self._busyCount == 0:
        self._coordinator.quit()