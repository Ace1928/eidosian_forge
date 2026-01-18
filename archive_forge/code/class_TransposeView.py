from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
class TransposeView(Table):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return itertranspose(self.source)