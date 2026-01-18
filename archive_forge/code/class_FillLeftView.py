from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
class FillLeftView(Table):

    def __init__(self, table, missing=None):
        self.table = table
        self.missing = missing

    def __iter__(self):
        return iterfillleft(self.table, self.missing)