from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class _LogStream(object):

    def __init__(self, ostream=None):
        self.ostream = ostream
        if self.ostream is not None:
            self.orig_stream_fileno = sys.stderr.fileno()

    def __enter__(self):
        if self.ostream is not None:
            self.orig_stream_dup = os.dup(self.orig_stream_fileno)
            os.dup2(self.ostream.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        if self.ostream is not None:
            os.close(self.orig_stream_fileno)
            os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
            os.close(self.orig_stream_dup)
            self.ostream.close()