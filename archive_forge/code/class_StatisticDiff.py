from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
class StatisticDiff:
    """
    Statistic difference on memory allocations between an old and a new
    Snapshot instance.
    """
    __slots__ = ('traceback', 'size', 'size_diff', 'count', 'count_diff')

    def __init__(self, traceback, size, size_diff, count, count_diff):
        self.traceback = traceback
        self.size = size
        self.size_diff = size_diff
        self.count = count
        self.count_diff = count_diff

    def __hash__(self):
        return hash((self.traceback, self.size, self.size_diff, self.count, self.count_diff))

    def __eq__(self, other):
        if not isinstance(other, StatisticDiff):
            return NotImplemented
        return self.traceback == other.traceback and self.size == other.size and (self.size_diff == other.size_diff) and (self.count == other.count) and (self.count_diff == other.count_diff)

    def __str__(self):
        text = '%s: size=%s (%s), count=%i (%+i)' % (self.traceback, _format_size(self.size, False), _format_size(self.size_diff, True), self.count, self.count_diff)
        if self.count:
            average = self.size / self.count
            text += ', average=%s' % _format_size(average, False)
        return text

    def __repr__(self):
        return '<StatisticDiff traceback=%r size=%i (%+i) count=%i (%+i)>' % (self.traceback, self.size, self.size_diff, self.count, self.count_diff)

    def _sort_key(self):
        return (abs(self.size_diff), self.size, abs(self.count_diff), self.count, self.traceback)