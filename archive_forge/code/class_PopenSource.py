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
class PopenSource(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @contextmanager
    def open(self, mode='r'):
        if not mode.startswith('r'):
            raise ArgumentError('source is read-only')
        self.kwargs['stdout'] = subprocess.PIPE
        proc = subprocess.Popen(*self.args, **self.kwargs)
        try:
            yield proc.stdout
        finally:
            pass