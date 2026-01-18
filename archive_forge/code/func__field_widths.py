import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
@staticmethod
def _field_widths(field_names, first_line):
    widths = [max(0, len(i) - 2) for i in first_line.split('+')[1:-1]]
    return dict(zip(field_names, widths))