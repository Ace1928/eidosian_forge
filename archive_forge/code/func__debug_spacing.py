from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def _debug_spacing(col_starts, row_starts):
    """Return a string suitable for inserting inside an <svg> tag that
    draws green lines where columns and rows start. This is very useful
    if you're developing this code and are debugging spacing issues.
    """
    t = ''
    for i, cs in enumerate(col_starts):
        t += f'<line id="cs-{i}" x1="{cs}" x2="{cs}" y1="0" y2="{row_starts[-1]}" stroke="green" stroke-width="1" />'
    for i, rs in enumerate(row_starts):
        t += f'<line id="rs-{i}" x1="0" x2="{col_starts[-1]}" y1="{rs}" y2="{rs}" stroke="green" stroke-width="1" />'
    return t