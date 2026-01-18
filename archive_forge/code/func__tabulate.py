import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
def _tabulate(self, info_list, col_widths, changed, order, bounds_dict):
    """
        Returns the supplied information as a table suitable for
        printing or paging.

        info_list:  List of the parameters name, type and mode.
        col_widths: Dictionary of column widths in characters
        changed:    List of parameters modified from their defaults.
        order:      The order of the table columns
        bound_dict: Dictionary of appropriately formatted bounds
        """
    contents, tail = ([], [])
    column_set = {k for _, row in info_list for k in row}
    columns = [col for col in order if col in column_set]
    title_row = []
    for i, col in enumerate(columns):
        width = col_widths[col] + 2
        col = col.capitalize()
        formatted = col.ljust(width) if i == 0 else col.center(width)
        title_row.append(formatted)
    contents.append(blue % ''.join(title_row) + '\n')
    for row, info in info_list:
        row_list = []
        for i, col in enumerate(columns):
            width = col_widths[col] + 2
            val = info[col] if col in info else ''
            formatted = val.ljust(width) if i == 0 else val.center(width)
            if col == 'bounds' and bounds_dict.get(row, False):
                mark_lbound, mark_ubound = bounds_dict[row]
                lval, uval = formatted.rsplit(',')
                lspace, lstr = lval.rsplit('(')
                ustr, uspace = uval.rsplit(')')
                lbound = lspace + '(' + cyan % lstr if mark_lbound else lval
                ubound = cyan % ustr + ')' + uspace if mark_ubound else uval
                formatted = f'{lbound},{ubound}'
            row_list.append(formatted)
        row_text = ''.join(row_list)
        if row in changed:
            row_text = red % row_text
        contents.append(row_text)
    return '\n'.join(contents + tail)