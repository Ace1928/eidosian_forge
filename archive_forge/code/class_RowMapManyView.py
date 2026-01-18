from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
class RowMapManyView(Table):

    def __init__(self, source, rowgenerator, header, failonerror=None):
        self.source = source
        self.rowgenerator = rowgenerator
        self.header = header
        self.failonerror = config.failonerror if failonerror is None else failonerror

    def __iter__(self):
        return iterrowmapmany(self.source, self.rowgenerator, self.header, self.failonerror)