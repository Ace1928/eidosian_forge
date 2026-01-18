import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def _unschedule(dt: float, _func: Callable) -> None:
    self.unschedule(_func)