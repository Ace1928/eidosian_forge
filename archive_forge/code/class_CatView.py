from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class CatView(Table):

    def __init__(self, sources, missing=None, header=None):
        self.sources = sources
        self.missing = missing
        self.header = header

    def __iter__(self):
        return itercat(self.sources, self.missing, self.header)