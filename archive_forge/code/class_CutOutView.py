from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class CutOutView(Table):

    def __init__(self, source, spec, missing=None):
        self.source = source
        self.spec = spec
        self.missing = missing

    def __iter__(self):
        return itercutout(self.source, self.spec, self.missing)