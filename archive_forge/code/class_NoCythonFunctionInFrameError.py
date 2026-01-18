from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class NoCythonFunctionInFrameError(CyGDBError):
    """
    raised when the user requests the current cython function, which is
    unavailable
    """
    msg = "Current function is a function cygdb doesn't know about"