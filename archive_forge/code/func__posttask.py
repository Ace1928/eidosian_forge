from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
def _posttask(self, key, value, dsk, state, id):
    t = default_timer()
    self._cache[key] = (self._metric(value), t)
    for k in state['released'] & self._cache.keys():
        metric, start = self._cache.pop(k)
        self.results.append(CacheData(k, dsk[k], metric, start, t))