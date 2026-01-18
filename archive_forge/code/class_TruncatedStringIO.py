from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class TruncatedStringIO(object):
    """Similar to io.StringIO, but can truncate the output by raising a
    StringTruncated exception"""

    def __init__(self, maxlen=None):
        self._val = ''
        self.maxlen = maxlen

    def write(self, data):
        if self.maxlen:
            if len(data) + len(self._val) > self.maxlen:
                self._val += data[0:self.maxlen - len(self._val)]
                raise StringTruncated()
        self._val += data

    def getvalue(self):
        return self._val