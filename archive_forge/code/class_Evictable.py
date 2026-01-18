import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
class Evictable:
    """Mixin for classes supporting the ``evict`` method."""
    Empty = Empty

    def evict(self) -> None:
        """Force evict until maxsize is enforced."""
        self._evict(range=count)

    def _evict(self, limit: int=100, range=range) -> None:
        try:
            [self._evict1() for _ in range(limit)]
        except IndexError:
            pass

    def _evict1(self) -> None:
        if self._evictcount <= self.maxsize:
            raise IndexError()
        try:
            self._pop_to_evict()
        except self.Empty:
            raise IndexError()