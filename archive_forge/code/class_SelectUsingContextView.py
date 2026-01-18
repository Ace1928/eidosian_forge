from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
class SelectUsingContextView(Table):

    def __init__(self, table, query):
        self.table = table
        self.query = query

    def __iter__(self):
        return iterselectusingcontext(self.table, self.query)