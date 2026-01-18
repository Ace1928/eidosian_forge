from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyGDBError(gdb.GdbError):
    """
    Base class for Cython-command related errors
    """

    def __init__(self, *args):
        args = args or (self.msg,)
        super(CyGDBError, self).__init__(*args)