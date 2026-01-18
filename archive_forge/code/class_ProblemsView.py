from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, Record
class ProblemsView(Table):

    def __init__(self, table, constraints, header):
        self.table = table
        self.constraints = constraints
        self.header = header

    def __iter__(self):
        return iterproblems(self.table, self.constraints, self.header)