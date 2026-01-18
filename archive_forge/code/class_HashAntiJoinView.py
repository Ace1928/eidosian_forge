from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
class HashAntiJoinView(Table):

    def __init__(self, left, right, lkey, rkey):
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey

    def __iter__(self):
        return iterhashantijoin(self.left, self.right, self.lkey, self.rkey)