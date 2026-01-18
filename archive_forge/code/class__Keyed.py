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
class _Keyed(namedtuple('Keyed', ['key', 'obj'])):

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __ne__(self, other):
        return self.key != other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key