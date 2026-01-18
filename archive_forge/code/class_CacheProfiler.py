from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
class CacheProfiler(Callback):
    """A profiler for dask execution at the scheduler cache level.

    Records the following information for each task:
        1. Key
        2. Task
        3. Size metric
        4. Cache entry time in seconds since the epoch
        5. Cache exit time in seconds since the epoch

    Examples
    --------

    >>> from operator import add, mul
    >>> from dask.threaded import get
    >>> from dask.diagnostics import CacheProfiler
    >>> dsk = {'x': 1, 'y': (add, 'x', 10), 'z': (mul, 'y', 2)}
    >>> with CacheProfiler() as prof:
    ...     get(dsk, 'z')
    22

    >>> prof.results    # doctest: +SKIP
    [CacheData(key='y', task=(add, 'x', 10), metric=1, cache_time=..., free_time=...),
     CacheData(key='z', task=(mul, 'y', 2), metric=1, cache_time=..., free_time=...)]

    The default is to count each task (``metric`` is 1 for all tasks). Other
    functions may used as a metric instead through the ``metric`` keyword. For
    example, the ``nbytes`` function found in ``cachey`` can be used to measure
    the number of bytes in the cache.

    >>> from cachey import nbytes                   # doctest: +SKIP
    >>> with CacheProfiler(metric=nbytes) as prof:  # doctest: +SKIP
    ...     get(dsk, 'z')
    22

    The profiling results can be visualized in a bokeh plot using the
    ``visualize`` method. Note that this requires bokeh to be installed.

    >>> prof.visualize() # doctest: +SKIP

    You can activate the profiler globally

    >>> prof.register()

    If you use the profiler globally you will need to clear out old results
    manually.

    >>> prof.clear()
    >>> prof.unregister()

    """

    def __init__(self, metric=None, metric_name=None):
        self.clear()
        self._metric = metric if metric else lambda value: 1
        if metric_name:
            self._metric_name = metric_name
        elif metric:
            self._metric_name = metric.__name__
        else:
            self._metric_name = 'count'

    def __enter__(self):
        self.clear()
        self.start_time = default_timer()
        return super().__enter__()

    def __exit__(self, *args):
        self.end_time = default_timer()
        return super().__exit__(*args)

    def _start(self, dsk):
        self._dsk.update(dsk)

    def _posttask(self, key, value, dsk, state, id):
        t = default_timer()
        self._cache[key] = (self._metric(value), t)
        for k in state['released'] & self._cache.keys():
            metric, start = self._cache.pop(k)
            self.results.append(CacheData(k, dsk[k], metric, start, t))

    def _finish(self, dsk, state, failed):
        t = default_timer()
        for k, (metric, start) in self._cache.items():
            self.results.append(CacheData(k, dsk[k], metric, start, t))
        self._cache.clear()

    def _plot(self, **kwargs):
        from dask.diagnostics.profile_visualize import plot_cache
        return plot_cache(self.results, self._dsk, self.start_time, self.end_time, self._metric_name, **kwargs)

    def visualize(self, **kwargs):
        """Visualize the profiling run in a bokeh plot.

        See also
        --------
        dask.diagnostics.profile_visualize.visualize
        """
        from dask.diagnostics.profile_visualize import visualize
        return visualize(self, **kwargs)

    def clear(self):
        """Clear out old results from profiler"""
        self.results = []
        self._cache = {}
        self._dsk = {}
        self.start_time = None
        self.end_time = None