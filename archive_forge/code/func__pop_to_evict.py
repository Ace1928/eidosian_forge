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
def _pop_to_evict(self):
    for _ in range(100):
        key = self._LRUkey()
        buf = self[key]
        try:
            buf.take()
        except (IndexError, self.Empty):
            self.pop(key)
        else:
            self.total -= 1
            if not len(buf):
                self.pop(key)
            else:
                self.move_to_end(key)
            break