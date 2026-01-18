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
def _do_a_fancy_diff(self, want, got, optionflags):
    if not optionflags & (REPORT_UDIFF | REPORT_CDIFF | REPORT_NDIFF):
        return False
    if optionflags & REPORT_NDIFF:
        return True
    return want.count('\n') > 2 and got.count('\n') > 2