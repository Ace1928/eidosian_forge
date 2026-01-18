from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class AddColumnView(Table):

    def __init__(self, table, field, col, index=None, missing=None):
        self._table = table
        self._field = field
        self._col = col
        self._index = index
        self._missing = missing

    def __iter__(self):
        return iteraddcolumn(self._table, self._field, self._col, self._index, self._missing)