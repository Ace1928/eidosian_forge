from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
class FieldMapView(Table):

    def __init__(self, source, mappings=None, failonerror=None, errorvalue=None):
        self.source = source
        if mappings is None:
            self.mappings = OrderedDict()
        else:
            self.mappings = mappings
        self.failonerror = config.failonerror if failonerror is None else failonerror
        self.errorvalue = errorvalue

    def __setitem__(self, key, value):
        self.mappings[key] = value

    def __iter__(self):
        return iterfieldmap(self.source, self.mappings, self.failonerror, self.errorvalue)