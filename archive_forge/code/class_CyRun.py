from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyRun(CythonExecutionControlCommand):
    """
    Run a Cython program. This is like the 'run' command, except that it
    displays Cython or Python source lines as well
    """
    name = 'cy run'
    invoke = libpython.dont_suppress_errors(CythonExecutionControlCommand.run)