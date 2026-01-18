from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
class HashRightJoinView(Table):

    def __init__(self, left, right, lkey, rkey, missing=None, cache=True, lprefix=None, rprefix=None):
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self.cache = cache
        self.llookup = None
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        if not self.cache or self.llookup is None:
            self.llookup = lookup(self.left, self.lkey)
        return iterhashrightjoin(self.left, self.right, self.lkey, self.rkey, self.missing, self.llookup, self.lprefix, self.rprefix)