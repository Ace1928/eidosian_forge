from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def _yield_all_rows(self, ws):
    for row in ws.get_all_values():
        yield tuple(row)