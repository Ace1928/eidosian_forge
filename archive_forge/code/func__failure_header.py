import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def _failure_header(self, test, example):
    out = [self.DIVIDER]
    if test.filename:
        if test.lineno is not None and example.lineno is not None:
            lineno = test.lineno + example.lineno + 1
        else:
            lineno = '?'
        out.append('File "%s", line %s, in %s' % (test.filename, lineno, test.name))
    else:
        out.append('Line %s, in %s' % (example.lineno + 1, test.name))
    out.append('Failed example:')
    source = example.source
    out.append(_indent(source))
    return '\n'.join(out)