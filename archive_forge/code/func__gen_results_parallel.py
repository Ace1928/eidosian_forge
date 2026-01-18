import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
def _gen_results_parallel(self, repeats, trial_fn, args):
    """Lazily generate results from an executor without submitting all jobs at once.
        """
    self._futures = deque()
    for r in repeats:
        if len(self._futures) < self.pre_dispatch:
            self._futures.append(self._executor.submit(trial_fn, r, *args))
            continue
        yield self._futures.popleft().result()
    while self._futures:
        yield self._futures.popleft().result()