from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
class FieldSelectView(Table):

    def __init__(self, source, field, where, complement=False, missing=None):
        self.source = source
        self.field = field
        self.where = where
        self.complement = complement
        self.missing = missing

    def __iter__(self):
        return iterfieldselect(self.source, self.field, self.where, self.complement, self.missing)