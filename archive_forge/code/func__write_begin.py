from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
def _write_begin(f, flds, lineterminator, caption, index_header, truncate):
    f.write("<table class='petl'>" + lineterminator)
    if caption is not None:
        f.write('<caption>%s</caption>' % caption + lineterminator)
    if flds:
        f.write('<thead>' + lineterminator)
        f.write('<tr>' + lineterminator)
        for i, h in enumerate(flds):
            if index_header:
                h = '%s|%s' % (i, h)
            if truncate:
                h = h[:truncate]
            f.write('<th>%s</th>' % h + lineterminator)
        f.write('</tr>' + lineterminator)
        f.write('</thead>' + lineterminator)
    f.write('<tbody>' + lineterminator)