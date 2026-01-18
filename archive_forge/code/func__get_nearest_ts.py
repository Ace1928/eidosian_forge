import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def _get_nearest_ts(self) -> float:
    """Get the nearest timestamp.

        Schedule from now, unless now is sufficiently close to last_ts, in
        which case use last_ts.  This clusters together scheduled items that
        probably want to be scheduled together.
        """
    last_ts = self.last_ts or self.next_ts
    ts = self.time()
    if ts - last_ts > 0.2:
        return ts
    return last_ts