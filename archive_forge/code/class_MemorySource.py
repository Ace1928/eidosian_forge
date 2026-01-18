from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
class MemorySource(object):
    """Memory data source. E.g.::

        >>> import petl as etl
        >>> data = b'foo,bar\\na,1\\nb,2\\nc,2\\n'
        >>> source = etl.MemorySource(data)
        >>> tbl = etl.fromcsv(source)
        >>> tbl
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' | '1' |
        +-----+-----+
        | 'b' | '2' |
        +-----+-----+
        | 'c' | '2' |
        +-----+-----+

        >>> sink = etl.MemorySource()
        >>> tbl.tojson(sink)
        >>> sink.getvalue()
        b'[{"foo": "a", "bar": "1"}, {"foo": "b", "bar": "2"}, {"foo": "c", "bar": "2"}]'

    Also supports appending.

    """

    def __init__(self, s=None):
        self.s = s
        self.buffer = None

    @contextmanager
    def open(self, mode='rb'):
        try:
            if 'r' in mode:
                if self.s is not None:
                    if 'b' in mode:
                        self.buffer = BytesIO(self.s)
                    else:
                        self.buffer = StringIO(self.s)
                else:
                    raise ArgumentError('no string data supplied')
            elif 'w' in mode:
                if self.buffer is not None:
                    self.buffer.close()
                if 'b' in mode:
                    self.buffer = BytesIO()
                else:
                    self.buffer = StringIO()
            elif 'a' in mode:
                if self.buffer is None:
                    if 'b' in mode:
                        self.buffer = BytesIO()
                    else:
                        self.buffer = StringIO()
            yield Uncloseable(self.buffer)
        except:
            raise
        finally:
            pass

    def getvalue(self):
        if self.buffer:
            return self.buffer.getvalue()