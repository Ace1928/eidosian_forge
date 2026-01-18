from __future__ import division, print_function, absolute_import
from petl.compat import next, string_types
from petl.util.base import iterpeek, ValuesView, Table
from petl.util.materialise import columns
class ArrayView(Table):

    def __init__(self, a):
        self.a = a

    def __iter__(self):
        yield tuple(self.a.dtype.names)
        for row in self.a:
            yield tuple(row)