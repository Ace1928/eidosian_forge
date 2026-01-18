import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
@staticmethod
def _assign_max_widths(x, max_width, min_width=0, fit_width=False):
    """Set maximum widths for columns of table `x`,
        with the last column recieving either leftover columns
        or `min_width`, depending on what offers more space.
        """
    if max_width > 0:
        term_width = max_width
    elif not _do_fit(fit_width):
        return
    else:
        term_width = utils.terminal_width()
        if not term_width:
            return
    field_count = len(x.field_names)
    try:
        first_line = x.get_string().splitlines()[0]
        if len(first_line) <= term_width:
            return
    except IndexError:
        return
    usable_total_width, optimal_width = TableFormatter._width_info(term_width, field_count)
    field_widths = TableFormatter._field_widths(x.field_names, first_line)
    shrink_fields, shrink_remaining = TableFormatter._build_shrink_fields(usable_total_width, optimal_width, field_widths, x.field_names)
    shrink_to = shrink_remaining // len(shrink_fields)
    for field in shrink_fields[:-1]:
        x.max_width[field] = max(min_width, shrink_to)
        shrink_remaining -= shrink_to
    field = shrink_fields[-1]
    x.max_width[field] = max(min_width, shrink_remaining)