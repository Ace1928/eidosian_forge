from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
class TailView(Table):

    def __init__(self, source, n):
        self.source = source
        self.n = n

    def __iter__(self):
        return itertail(self.source, self.n)