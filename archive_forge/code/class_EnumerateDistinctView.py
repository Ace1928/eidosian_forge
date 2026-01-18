from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
class EnumerateDistinctView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        offset, multiplier = self.autoincrement
        yield ('id', self.value)
        for n, (v, _) in enumerate(rowgroupby(self.table, self.value)):
            yield (n * multiplier + offset, v)