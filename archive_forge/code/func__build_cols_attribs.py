from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_cols_attribs(props, path):
    parts = path.split('/')
    row = parts[-1]
    fmt = '{0}="{{{0}}}"'
    cols = [fmt.format(e) for e in props]
    atts = ' '.join(cols)
    res = ' <{0} {1} />\n'.format(row, atts)
    return res