from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
class IntervalTreeLookup(object):

    def __init__(self, tree, include_stop=False):
        self.tree = tree
        self.include_stop = include_stop

    def search(self, start, stop=None):
        results = _search_tree(self.tree, start, stop, self.include_stop)
        return [r.data for r in results]
    find = search