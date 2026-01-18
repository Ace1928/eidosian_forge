from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
@contextmanager
def _get_hdf5_file(source, mode='r'):
    import tables
    needs_closing = False
    if isinstance(source, string_types):
        h5file = tables.open_file(source, mode=mode)
        needs_closing = True
    elif isinstance(source, tables.File):
        h5file = source
    else:
        raise ArgumentError('invalid source argument, expected file name or tables.File object, found: %r' % source)
    try:
        yield h5file
    finally:
        if needs_closing:
            h5file.close()