from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyExec(CythonCommand, libpython.PyExec, EvaluateOrExecuteCodeMixin):
    """
    Execute Python code in the nearest Python or Cython frame.
    """
    name = '-cy-exec'
    command_class = gdb.COMMAND_STACK
    completer_class = gdb.COMPLETE_NONE

    @libpython.dont_suppress_errors
    def invoke(self, expr, from_tty):
        expr, input_type = self.readcode(expr)
        executor = libpython.PythonCodeExecutor()
        executor.xdecref(self.evalcode(expr, executor.Py_file_input))