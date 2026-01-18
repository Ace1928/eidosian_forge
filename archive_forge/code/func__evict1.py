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
def _evict1(self) -> None:
    if self._evictcount <= self.maxsize:
        raise IndexError()
    try:
        self._pop_to_evict()
    except self.Empty:
        raise IndexError()