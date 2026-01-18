from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class AddFieldsView(Table):

    def __init__(self, source, field_defs, missing=None):
        self.source = stack(source, missing=missing)
        self.field_defs = field_defs

    def __iter__(self):
        return iteraddfields(self.source, self.field_defs)