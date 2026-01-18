from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def _compare_grouped_stats(old_group, new_group):
    statistics = []
    for traceback, stat in new_group.items():
        previous = old_group.pop(traceback, None)
        if previous is not None:
            stat = StatisticDiff(traceback, stat.size, stat.size - previous.size, stat.count, stat.count - previous.count)
        else:
            stat = StatisticDiff(traceback, stat.size, stat.size, stat.count, stat.count)
        statistics.append(stat)
    for traceback, stat in old_group.items():
        stat = StatisticDiff(traceback, 0, -stat.size, 0, -stat.count)
        statistics.append(stat)
    return statistics