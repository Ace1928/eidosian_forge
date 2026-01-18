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
class AntiJoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.left = left
            self.right = right
        else:
            self.left = sort(left, lkey, buffersize=buffersize, tempdir=tempdir, cache=cache)
            self.right = sort(right, rkey, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.lkey = lkey
        self.rkey = rkey

    def __iter__(self):
        return iterantijoin(self.left, self.right, self.lkey, self.rkey)