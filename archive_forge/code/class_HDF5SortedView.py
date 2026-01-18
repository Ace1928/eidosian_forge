from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
class HDF5SortedView(Table):

    def __init__(self, source, where=None, name=None, sortby=None, checkCSI=False, start=None, stop=None, step=None):
        self.source = source
        self.where = where
        self.name = name
        self.sortby = sortby
        self.checkCSI = checkCSI
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return iterhdf5sorted(self.source, self.where, self.name, self.sortby, self.checkCSI, self.start, self.stop, self.step)