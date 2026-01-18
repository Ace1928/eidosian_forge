from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
class Statistic:
    """
    Statistic difference on memory allocations between two Snapshot instance.
    """
    __slots__ = ('traceback', 'size', 'count')

    def __init__(self, traceback, size, count):
        self.traceback = traceback
        self.size = size
        self.count = count

    def __hash__(self):
        return hash((self.traceback, self.size, self.count))

    def __eq__(self, other):
        if not isinstance(other, Statistic):
            return NotImplemented
        return self.traceback == other.traceback and self.size == other.size and (self.count == other.count)

    def __str__(self):
        text = '%s: size=%s, count=%i' % (self.traceback, _format_size(self.size, False), self.count)
        if self.count:
            average = self.size / self.count
            text += ', average=%s' % _format_size(average, False)
        return text

    def __repr__(self):
        return '<Statistic traceback=%r size=%i count=%i>' % (self.traceback, self.size, self.count)

    def _sort_key(self):
        return (self.size, self.count, self.traceback)