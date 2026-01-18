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
class CompressedSource(object):
    """Handle IO from a file-like object and (de)compress with a codec
    
    The `source` argument (source class) is the source class that will handle
    the actual input/output stream. E.g: :class:`petl.io.sources.URLSource`.
    
    The `codec` argument (source class) is the source class that will handle
    the (de)compression of the stream. E.g: :class:`petl.io.sources.GzipSource`.
    """

    def __init__(self, source, codec):
        self.source = source
        self.codec = codec

    @contextmanager
    def open(self, mode='rb'):
        with self.source.open(mode=mode) as filehandle:
            transcoder = self.codec(filehandle)
            with transcoder.open(mode=mode) as stream:
                yield stream