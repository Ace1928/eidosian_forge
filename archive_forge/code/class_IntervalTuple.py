from math import isinf, isnan
import itertools
import numpy
class IntervalTuple(object):

    def __init__(self, values):
        self.values = tuple(values)

    def union(self, other):
        if isinstance(other, Interval):
            return UNKNOWN_TUPLE_RANGE
        return IntervalTuple((x.union(y) for x, y in zip(self.values, other.values)))

    def intersect(self, other):
        if isinstance(other, Interval):
            return UNKNOWN_TUPLE_RANGE
        return IntervalTuple((x.intersect(y) for x, y in zip(self.values, other.values)))

    @property
    def high(self):
        return UNKNOWN_RANGE.high

    @property
    def low(self):
        return UNKNOWN_RANGE.low

    def __getitem__(self, index):
        out = None
        low = max(0, index.low)
        high = min(len(self.values) - 1, index.high)
        for i in range(low, high + 1):
            if out is None:
                out = self.values[i]
            else:
                out = out.union(self.values[i])
        return out or UNKNOWN_RANGE

    def widen(self, other):
        if isinstance(other, Interval):
            return UNKNOWN_TUPLE_RANGE
        return IntervalTuple((s.widen(o) for s, o in zip(self.values, other.values)))

    def __add__(self, other):
        if isinstance(other, Interval):
            return UNKNOWN_TUPLE_RANGE
        return IntervalTuple(self.values + other.values)