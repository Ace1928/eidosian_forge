import sys
import traceback
from mako import compat
from mako import util
def _get_reformatted_records(self, records):
    for rec in records:
        if rec[6] is not None:
            yield (rec[4], rec[5], rec[2], rec[6])
        else:
            yield tuple(rec[0:4])