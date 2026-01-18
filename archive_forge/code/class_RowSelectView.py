from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
class RowSelectView(Table):

    def __init__(self, source, where, missing=None, complement=False):
        self.source = source
        self.where = where
        self.missing = missing
        self.complement = complement

    def __iter__(self):
        return iterrowselect(self.source, self.where, self.missing, self.complement)