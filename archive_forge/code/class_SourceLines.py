from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from contextlib import closing
import copy
import inspect
import os
import re
import sys
import textwrap
from io import StringIO
import numba.core.dispatcher
from numba.core import ir
class SourceLines(Mapping):

    def __init__(self, func):
        try:
            lines, startno = inspect.getsourcelines(func)
        except OSError:
            self.lines = ()
            self.startno = 0
        else:
            self.lines = textwrap.dedent(''.join(lines)).splitlines()
            self.startno = startno

    def __getitem__(self, lineno):
        try:
            return self.lines[lineno - self.startno].rstrip()
        except IndexError:
            return ''

    def __iter__(self):
        return iter((self.startno + i for i in range(len(self.lines))))

    def __len__(self):
        return len(self.lines)

    @property
    def avail(self):
        return bool(self.lines)