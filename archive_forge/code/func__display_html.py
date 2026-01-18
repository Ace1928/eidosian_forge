from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def _display_html(table, limit=0, vrepr=None, index_header=None, caption=None, tr_style=None, td_styles=None, encoding=None, truncate=None, epilogue=None):
    if limit == 0:
        limit = config.display_limit
    if vrepr is None:
        vrepr = config.display_vrepr
    if index_header is None:
        index_header = config.display_index_header
    if encoding is None:
        encoding = locale.getpreferredencoding()
    table, overflow = _vis_overflow(table, limit)
    buf = MemorySource()
    tohtml(table, buf, encoding=encoding, index_header=index_header, vrepr=vrepr, caption=caption, tr_style=tr_style, td_styles=td_styles, truncate=truncate)
    output = text_type(buf.getvalue(), encoding)
    if epilogue:
        output += '<p>%s</p>' % epilogue
    elif overflow:
        output += '<p><strong>...</strong></p>'
    return output