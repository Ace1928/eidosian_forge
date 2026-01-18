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
def _mergesorted(key=None, reverse=False, *iterables):
    if reverse:
        return _shortlistmergesorted(key, True, *iterables)
    else:
        return _heapqmergesorted(key, *iterables)