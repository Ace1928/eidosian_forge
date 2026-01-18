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
class MergeDuplicatesView(Table):

    def __init__(self, table, key, missing=None, presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.table = table
        else:
            self.table = sort(table, key, buffersize=buffersize, tempdir=tempdir, cache=cache)
        self.key = key
        self.missing = missing

    def __iter__(self):
        return itermergeduplicates(self.table, self.key, self.missing)