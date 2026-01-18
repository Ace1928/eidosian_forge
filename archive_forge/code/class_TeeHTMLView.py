from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
class TeeHTMLView(Table):

    def __init__(self, table, source=None, encoding=None, errors='strict', caption=None, vrepr=text_type, lineterminator='\n', index_header=False, tr_style=None, td_styles=None, truncate=None):
        self.table = table
        self.source = source
        self.encoding = encoding
        self.errors = errors
        self.caption = caption
        self.vrepr = vrepr
        self.lineterminator = lineterminator
        self.index_header = index_header
        self.tr_style = tr_style
        self.td_styles = td_styles
        self.truncate = truncate

    def __iter__(self):
        table = self.table
        source = self.source
        encoding = self.encoding
        errors = self.errors
        lineterminator = self.lineterminator
        caption = self.caption
        index_header = self.index_header
        tr_style = self.tr_style
        td_styles = self.td_styles
        vrepr = self.vrepr
        truncate = self.truncate
        with source.open('wb') as buf:
            if PY2:
                codec = getcodec(encoding)
                f = codec.streamwriter(buf, errors=errors)
            else:
                f = io.TextIOWrapper(buf, encoding=encoding, errors=errors, newline='')
            try:
                it = iter(table)
                try:
                    hdr = next(it)
                    yield hdr
                except StopIteration:
                    hdr = []
                _write_begin(f, hdr, lineterminator, caption, index_header, truncate)
                if tr_style and callable(tr_style):
                    it = (Record(row, hdr) for row in it)
                for row in it:
                    _write_row(f, hdr, row, lineterminator, vrepr, tr_style, td_styles, truncate)
                    yield row
                _write_end(f, lineterminator)
                f.flush()
            finally:
                if not PY2:
                    f.detach()