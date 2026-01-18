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
class SortView(Table):

    def __init__(self, source, key=None, reverse=False, buffersize=None, tempdir=None, cache=True):
        self.source = source
        self.key = key
        self.reverse = reverse
        if buffersize is None:
            self.buffersize = config.sort_buffersize
        else:
            self.buffersize = buffersize
        self.tempdir = tempdir
        self.cache = cache
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def clearcache(self):
        debug('clear cache')
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def __iter__(self):
        source = self.source
        key = self.key
        reverse = self.reverse
        if self.cache and self._memcache is not None:
            return self._iterfrommemcache()
        elif self.cache and self._filecache is not None:
            return self._iterfromfilecache()
        else:
            return self._iternocache(source, key, reverse)

    def _iterfrommemcache(self):
        debug('iterate from memory cache')
        yield tuple(self._hdrcache)
        for row in self._memcache:
            yield tuple(row)

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

    def _iternocache(self, source, key, reverse):
        debug('iterate without cache')
        self.clearcache()
        it = iter(source)
        try:
            hdr = next(it)
        except StopIteration:
            if key is None:
                return
            hdr = []
        yield tuple(hdr)
        if key is not None:
            indices = asindices(hdr, key)
        else:
            indices = range(len(hdr))
        getkey = comparable_itemgetter(*indices)
        rows = list(itertools.islice(it, 0, self.buffersize))
        rows.sort(key=getkey, reverse=reverse)
        if self.buffersize is None or len(rows) < self.buffersize:
            if self.cache:
                debug('caching mem')
                self._hdrcache = hdr
                self._memcache = rows
                self._getkey = getkey
            for row in rows:
                yield tuple(row)
        else:
            chunkfiles = []
            while rows:
                with NamedTemporaryFile(dir=self.tempdir, delete=False, mode='wb') as f:
                    wrapper = _NamedTempFileDeleteOnGC(f.name)
                    debug('created temporary chunk file %s' % f.name)
                    for row in rows:
                        pickle.dump(row, f, protocol=-1)
                    f.flush()
                    chunkfiles.append(wrapper)
                rows = list(itertools.islice(it, 0, self.buffersize))
                rows.sort(key=getkey, reverse=reverse)
            if self.cache:
                debug('caching files')
                self._hdrcache = hdr
                self._filecache = chunkfiles
                self._getkey = getkey
            chunkiters = [_iterchunk(f.name) for f in chunkfiles]
            for row in _mergesorted(getkey, reverse, *chunkiters):
                yield tuple(row)