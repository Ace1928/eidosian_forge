from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def _yield_by_range(self, ws):
    found = ws.get_values(self.cell_range)
    for row in found:
        yield tuple(row)