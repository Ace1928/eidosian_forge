from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonInfo(CythonBase, libpython.PythonInfo):
    """
    Implementation of the interface dictated by libpython.LanguageInfo.
    """

    def lineno(self, frame):
        if self.is_cython_function(frame):
            return self.get_cython_lineno(frame)[1]
        return super(CythonInfo, self).lineno(frame)

    def get_source_line(self, frame):
        try:
            line = super(CythonInfo, self).get_source_line(frame)
        except gdb.GdbError:
            return None
        else:
            return line.strip() or None

    def exc_info(self, frame):
        if self.is_python_function:
            return super(CythonInfo, self).exc_info(frame)

    def runtime_break_functions(self):
        if self.is_cython_function():
            return self.get_cython_function().step_into_functions
        return ()

    def static_break_functions(self):
        result = ['PyEval_EvalFrameEx']
        result.extend(self.cy.functions_by_cname)
        return result