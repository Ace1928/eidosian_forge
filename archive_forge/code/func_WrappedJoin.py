from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting_windows  # pylint: disable=unused-import
import termcolor
def WrappedJoin(items, separator=' | ', width=80):
    """Joins the items by the separator, wrapping lines at the given width."""
    lines = []
    current_line = ''
    for index, item in enumerate(items):
        is_final_item = index == len(items) - 1
        if is_final_item:
            if len(current_line) + len(item) <= width:
                current_line += item
            else:
                lines.append(current_line.rstrip())
                current_line = item
        elif len(current_line) + len(item) + len(separator) <= width:
            current_line += item + separator
        else:
            lines.append(current_line.rstrip())
            current_line = item + separator
    lines.append(current_line)
    return lines