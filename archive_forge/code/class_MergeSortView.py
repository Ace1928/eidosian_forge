from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices
class MergeSortView(Table):

    def __init__(self, tables, key=None, reverse=False, presorted=False, missing=None, header=None, buffersize=None, tempdir=None, cache=True):
        self.key = key
        if presorted:
            self.tables = tables
        else:
            self.tables = [sort(t, key=key, reverse=reverse, buffersize=buffersize, tempdir=tempdir, cache=cache) for t in tables]
        self.missing = missing
        self.header = header
        self.reverse = reverse

    def __iter__(self):
        return itermergesort(self.tables, self.key, self.header, self.missing, self.reverse)