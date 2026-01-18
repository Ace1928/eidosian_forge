from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
def _writetext(table, source, mode, encoding, errors, template, prologue, epilogue):
    assert template is not None, 'template is required'
    source = write_source_from_arg(source, mode)
    with source.open(mode) as buf:
        if PY2:
            codec = getcodec(encoding)
            f = codec.streamwriter(buf, errors=errors)
        else:
            f = io.TextIOWrapper(buf, encoding=encoding, errors=errors, newline='')
        try:
            if prologue is not None:
                f.write(prologue)
            it = iter(table)
            try:
                hdr = next(it)
            except StopIteration:
                hdr = []
            flds = list(map(text_type, hdr))
            for row in it:
                rec = asdict(flds, row)
                s = template.format(**rec)
                f.write(s)
            if epilogue is not None:
                f.write(epilogue)
            f.flush()
        finally:
            if not PY2:
                f.detach()