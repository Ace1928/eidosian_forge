from __future__ import absolute_import, print_function, division
import itertools
import operator
from collections import OrderedDict
from petl.compat import next, string_types, reduce, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, rowgroupby
from petl.util.base import values
from petl.util.counting import nrows
from petl.transform.sorts import sort, mergesort
from petl.transform.basics import cut
from petl.transform.dedup import distinct
class MultiAggregateView(Table):

    def __init__(self, source, key, aggregation=None, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted or key is None:
            self.source = source
        else:
            self.source = sort(source, key, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.key = key
        if aggregation is None:
            self.aggregation = OrderedDict()
        elif isinstance(aggregation, (list, tuple)):
            self.aggregation = OrderedDict()
            for t in aggregation:
                self.aggregation[t[0]] = t[1:]
        elif isinstance(aggregation, dict):
            self.aggregation = aggregation
        else:
            raise ArgumentError('expected aggregation is None, list, tuple or dict, found %r' % aggregation)

    def __iter__(self):
        return itermultiaggregate(self.source, self.key, self.aggregation)

    def __setitem__(self, key, value):
        self.aggregation[key] = value