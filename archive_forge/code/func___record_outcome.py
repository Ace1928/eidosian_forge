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
def __record_outcome(self, test, f, t):
    """
        Record the fact that the given DocTest (`test`) generated `f`
        failures out of `t` tried examples.
        """
    f2, t2 = self._name2ft.get(test.name, (0, 0))
    self._name2ft[test.name] = (f + f2, t + t2)
    self.failures += f
    self.tries += t