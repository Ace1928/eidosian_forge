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
def __patched_linecache_getlines(self, filename, module_globals=None):
    m = self.__LINECACHE_FILENAME_RE.match(filename)
    if m and m.group('name') == self.test.name:
        example = self.test.examples[int(m.group('examplenum'))]
        return example.source.splitlines(keepends=True)
    else:
        return self.save_linecache_getlines(filename, module_globals)