from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
class FillDownView(Table):

    def __init__(self, table, fields, missing=None):
        self.table = table
        self.fields = fields
        self.missing = missing

    def __iter__(self):
        return iterfilldown(self.table, self.fields, self.missing)