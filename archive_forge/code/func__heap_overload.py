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
@property
def _heap_overload(self):
    """Compute how much is heap bigger than data [percents]."""
    return len(self._heap) * 100 / max(len(self._data), 1) - 100