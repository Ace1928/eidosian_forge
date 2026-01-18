from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_cols_inline(props, path, is_value, use_tag):
    parts = path.split('/')
    if use_tag:
        if len(parts) < 2:
            raise ValueError("Tag not in format 'row/col': %s" % path)
        col = parts[-1]
        row = parts[-2:-1][0]
    else:
        col = '{0}'
        row = parts[-1]
    fld = '{{{0}}}' if is_value else '{0}'
    fmt = '<{0}>{1}</{0}>'.format(col, fld)
    cols = [fmt.format(e) for e in props]
    tags = ''.join(cols)
    res = ' <{0}>{1}</{0}>\n'.format(row, tags)
    return res