from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyEval(gdb.Function, CythonBase, EvaluateOrExecuteCodeMixin):
    """
    Evaluate Python code in the nearest Python or Cython frame and return
    """

    @libpython.dont_suppress_errors
    @gdb_function_value_to_unicode
    def invoke(self, python_expression):
        input_type = libpython.PythonCodeExecutor.Py_eval_input
        return self.evalcode(python_expression, input_type)