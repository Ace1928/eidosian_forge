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
def _iterfromfilecache(self):
    filecache = self._filecache
    filenames = list(map(operator.attrgetter('name'), filecache))
    debug('iterate from file cache: %r', filenames)
    yield tuple(self._hdrcache)
    chunkiters = [_iterchunk(fn) for fn in filenames]
    rows = _mergesorted(self._getkey, self.reverse, *chunkiters)
    try:
        for row in rows:
            yield tuple(row)
    finally:
        debug('attempt cleanup from generator')
        del chunkiters
        del rows
        del filecache
        debug('exiting generator')