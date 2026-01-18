from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
class SplitView(Table):

    def __init__(self, source, field, pattern, newfields=None, include_original=False, maxsplit=0, flags=0):
        self.source = source
        self.field = field
        self.pattern = pattern
        self.newfields = newfields
        self.include_original = include_original
        self.maxsplit = maxsplit
        self.flags = flags

    def __iter__(self):
        return itersplit(self.source, self.field, self.pattern, self.newfields, self.include_original, self.maxsplit, self.flags)