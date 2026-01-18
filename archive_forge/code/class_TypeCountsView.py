from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
class TypeCountsView(Table):

    def __init__(self, table, field):
        self.table = table
        self.field = field

    def __iter__(self):
        counter = typecounter(self.table, self.field)
        yield ('type', 'count', 'frequency')
        counts = counter.most_common()
        total = sum((c[1] for c in counts))
        for c in counts:
            yield (c[0], c[1], float(c[1]) / total)