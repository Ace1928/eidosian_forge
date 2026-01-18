from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
class RowMapView(Table):

    def __init__(self, source, rowmapper, header, failonerror=None):
        self.source = source
        self.rowmapper = rowmapper
        self.header = header
        self.failonerror = config.failonerror if failonerror is None else failonerror

    def __iter__(self):
        return iterrowmap(self.source, self.rowmapper, self.header, self.failonerror)