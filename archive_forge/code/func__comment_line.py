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
def _comment_line(line):
    """Return a commented form of the given line"""
    line = line.rstrip()
    if line:
        return '# ' + line
    else:
        return '#'