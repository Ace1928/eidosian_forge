from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
class SetHeaderView(Table):

    def __init__(self, source, header):
        self.source = source
        self.header = header

    def __iter__(self):
        return itersetheader(self.source, self.header)