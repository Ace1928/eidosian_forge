import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
@staticmethod
def _width_info(term_width, field_count):
    usable_total_width = max(0, term_width - 1 - 3 * field_count)
    if field_count == 0:
        optimal_width = 0
    else:
        optimal_width = max(0, usable_total_width // field_count)
    return (usable_total_width, optimal_width)