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
class ConvertToIncrementingCounterView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        table = itertools.chain([hdr], it)
        value = self.value
        vidx = hdr.index(value)
        outhdr = list(hdr)
        outhdr[vidx] = '%s_id' % value
        yield tuple(outhdr)
        offset, multiplier = self.autoincrement
        for n, (_, group) in enumerate(rowgroupby(table, value)):
            for row in group:
                outrow = list(row)
                outrow[vidx] = n * multiplier + offset
                yield tuple(outrow)