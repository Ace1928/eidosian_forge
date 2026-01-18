import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
def _cancel_futures(self):
    if self._executor is not None:
        for f in self._futures:
            f.cancel()