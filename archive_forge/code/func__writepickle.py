from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def _writepickle(table, source, mode, protocol, write_header):
    source = write_source_from_arg(source, mode)
    with source.open(mode) as f:
        it = iter(table)
        try:
            hdr = next(it)
        except StopIteration:
            return
        if write_header:
            pickle.dump(hdr, f, protocol)
        for row in it:
            pickle.dump(row, f, protocol)