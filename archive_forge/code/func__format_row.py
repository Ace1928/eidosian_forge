import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
def _format_row(row):
    new_row = []
    for r in row:
        if isinstance(r, columns.FormattableColumn):
            r = r.human_readable()
        if isinstance(r, str):
            r = r.replace('\r\n', '\n').replace('\r', ' ')
        new_row.append(r)
    return new_row