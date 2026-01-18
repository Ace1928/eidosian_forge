from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
class FieldConvertView(Table):

    def __init__(self, source, converters=None, failonerror=None, errorvalue=None, where=None, pass_row=False):
        self.source = source
        if converters is None:
            self.converters = dict()
        elif isinstance(converters, dict):
            self.converters = converters
        elif isinstance(converters, (tuple, list)):
            self.converters = dict([(i, v) for i, v in enumerate(converters)])
        else:
            raise ArgumentError('unexpected converters: %r' % converters)
        self.failonerror = config.failonerror if failonerror is None else failonerror
        self.errorvalue = errorvalue
        self.where = where
        self.pass_row = pass_row

    def __iter__(self):
        return iterfieldconvert(self.source, self.converters, self.failonerror, self.errorvalue, self.where, self.pass_row)

    def __setitem__(self, key, value):
        self.converters[key] = value