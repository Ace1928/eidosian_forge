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
class CrossJoinView(Table):

    def __init__(self, *sources, **kwargs):
        self.sources = sources
        self.prefix = kwargs.get('prefix', False)

    def __iter__(self):
        return itercrossjoin(self.sources, self.prefix)